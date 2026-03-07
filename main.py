import os
import sys
import time
import shutil
import csv
import random
import yaml
import numpy as np
import torch
from scipy.spatial.distance import mahalanobis

# ASE Imports
from ase.io import read, write
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.calculators.vasp import Vasp
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

# SevenNet Imports
from sevenn.sevennet_calculator import SevenNetCalculator
from sevenn import __version__ as sevenn_version
import sevenn._keys as KEY
from sevenn.parse_input import read_config_yaml
from sevenn.scripts.train import train_v2
from sevenn.sevenn_logger import Logger
from sevenn.util import unique_filepath

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # File Paths
    'config_yaml': 'config.yaml',
    'model_name': 'checkpoint_best.pth',
    'initial_trainset': 'cubic1_200.extxyz',
    'initial_structure': 'POSCAR',
    'trainset_file': 'trainset.extxyz',
    'total_trainset_file': 'total_trainset.extxyz',
    'csv_output': 'result.csv',
    'log_file': 'log.sevenn',
    
    # Active Learning Parameters
    'species': ['Cs', 'Pb', 'Br'],
    'threshold': 5,
    'num_stru_train': 30,
    'k_sample_ratio': 0.2,
    'sample_interval': 10,
    'skip_iter': np.inf,
    'device': 'cpu',

    # MD Parameters
    'ensemble': "NVT",  # Options: "NVT", "NVE"
    'total_steps': 5000,
    'timestep_fs': 1.0,
    'temp_begin': 300,
    'temp_end': 300,  # Currently unused but kept for compatibility
    'save_interval': 1,
    'friction': 0.04,
}

VASP_SETTINGS = {
    "command": "srun --mpi=pmi2 vasp_std",
    "gga": "PE",
    "directory": './vasp',
    "ncore": 64,
    "kpar": 8,
    "kpts": [2, 2, 2],
    "gamma": True,
    "istart": 0,
    "icharge": 2,
    "nwrite": 2,
    "encut": 380,
    "nelm": 200,
    "ediff": 0.1E-05,
    "nelmin": 4,
    "algo": "Normal",
    "ibrion": -1,
    "isif": 2,
    "isym": 0,
    "prec": "Accurate",
    "lreal": "Auto",
    "lwave": False,
    "lcharg": False,
    "ismear": 0,
    "sigma": 0.05
}

# =============================================================================
# Helper Classes
# =============================================================================

class MahalanobisCalculator:
    """
    Calculates the Mahalanobis distance for uncertainty estimation 
    based on atomic embeddings from the SevenNet model.
    """
    def __init__(self, dataset, species, model_path, device='cpu'):
        self.data = dataset
        self.species = species
        self.model = SevenNetCalculator(model=model_path, device=device)
        self.ebd_dict = {ele: [] for ele in self.species}
        self.avg = {}
        self.sigma = {}
        self.inv_sig = {}

    def init_statistics(self):
        """Initialize mean and covariance matrix from the dataset."""
        print(f"Initializing query parameters with {len(self.data)} structures...")
        
        # Collect embeddings
        for atom in self.data:
            ebd_list = self.model.get_property('embeddings', atom)
            element_symbols = np.array(atom.get_chemical_symbols())
            for element in self.species:
                mask = (element_symbols == element)
                if np.any(mask):
                    self.ebd_dict[element].append(ebd_list[mask])

        # Calculate statistics
        for element in self.species:
            if self.ebd_dict[element]:
                self.ebd_dict[element] = np.vstack(self.ebd_dict[element])
                self.avg[element] = np.mean(self.ebd_dict[element], axis=0)
                self.sigma[element] = np.cov(self.ebd_dict[element], rowvar=False, ddof=True)
                # Use pseudo-inverse for stability
                self.inv_sig[element] = np.linalg.pinv(self.sigma[element])
            else:
                print(f"Warning: No embeddings found for species {element}")

    def calculate_distance(self, atom):
        """Calculate the average Mahalanobis distance for the given structure."""
        ebds = self.model.get_property('embeddings', atom)
        element_symbols = np.array(atom.get_chemical_symbols())
        result = []

        for element in self.species:
            mask = (element_symbols == element)
            if not np.any(mask):
                continue
                
            ebd_element = ebds[mask]
            # Calculate distance for each atom of this element
            dists = [mahalanobis(v, self.avg[element], self.inv_sig[element]) 
                     for v in ebd_element]
            
            if dists:
                result.append(np.mean(dists))

        return np.mean(result) if result else 0.0


# =============================================================================
# Helper Functions
# =============================================================================

def setup_sevenn_config(config_path, log_file):
    """Parses SevenNet configuration and sets up logging."""
    global_config = {
        'version': sevenn_version,
        'when': time.ctime(),
        '_model_type': 'E3_equivariant_model',
    }
    
    # Logger setup
    log_fname = unique_filepath(f'{os.path.abspath(".")}/{log_file}')
    # Note: Logger usage here is slightly adapted to just setup config, 
    # actual logging is done via print/csv in this script mostly.
    
    try:
        model_config, train_config, data_config = read_config_yaml(config_path, return_separately=True)
    except Exception as e:
        print(f'Failed to parse {config_path}: {e}')
        sys.exit(1)

    # Update global config
    train_config[KEY.IS_DDP] = False
    train_config[KEY.DDP_BACKEND] = 'nccl'
    train_config[KEY.LOCAL_RANK] = 0
    train_config[KEY.RANK] = 0
    train_config[KEY.WORLD_SIZE] = 1
    
    global_config.update(model_config)
    global_config.update(train_config)
    global_config.update(data_config)

    # Set seeds
    seed = global_config.get(KEY.RANDOM_SEED, 42)
    random.seed(seed)
    torch.manual_seed(seed)
    
    return global_config, log_fname

def sample_structures(filename, k):
    """Randomly samples a fraction k of structures from an extxyz file."""
    structures = read(filename, index=':')
    sample_size = max(1, int(len(structures) * k))
    return random.sample(structures, sample_size)

def get_dft_calculator():
    """Returns the VASP calculator instance."""
    return Vasp(**VASP_SETTINGS)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # 1. Setup Data and Config
    sevenn_config, log_fname = setup_sevenn_config(CONFIG['config_yaml'], CONFIG['log_file'])
    
    # Initialize datasets
    trainset_atoms = read(CONFIG['initial_trainset'], index=':')
    print(f"Initial training set size: {len(trainset_atoms)}")
    
    # Reset/Create files
    open(CONFIG['trainset_file'], 'w').close() # Clear file
    write(CONFIG['total_trainset_file'], trainset_atoms)
    
    # 2. Setup MD System
    atoms = read(CONFIG['initial_structure'])
    MaxwellBoltzmannDistribution(atoms, temperature_K=CONFIG['temp_begin'], force_temp=True)
    Stationary(atoms)

    # Setup Dynamics
    if CONFIG['ensemble'] == 'NVE':
        dyn = VelocityVerlet(atoms, timestep=CONFIG['timestep_fs'] * units.fs)
    elif CONFIG['ensemble'] == 'NVT':
        dyn = Langevin(atoms,
                       timestep=CONFIG['timestep_fs'] * units.fs,
                       temperature_K=CONFIG['temp_begin'],
                       friction=CONFIG['friction'])
    else:
        raise ValueError(f"Ensemble {CONFIG['ensemble']} unavailable!")

    # 3. Setup Logging and Output
    csv_file = open(CONFIG['csv_output'], 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'epot', 'ekin', 'temp'])

    # MD Observers
    def correct_result():
        # Preserve VASP results format for ASE
        if atoms.calc:
             atoms.calc.results = {
                 'energy': atoms.calc.results.get('energy'),
                 'forces': atoms.get_forces(),
                 'stress': atoms.get_stress()
             }

    def print_status():
        step = dyn.get_number_of_steps()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        print(f'Step: {step:5d}, E_pot: {epot:10.4f} eV, '
              f'E_kin: {ekin:10.4f} eV, Temp: {temp:7.1f} K')
        csv_writer.writerow([step, epot, ekin, temp])

    def save_trajectory():
        write('md_result.extxyz', atoms, append=True)

    dyn.attach(correct_result, interval=1)
    dyn.attach(print_status, interval=1)
    dyn.attach(save_trajectory, interval=CONFIG['save_interval'])

    # 4. Active Learning Loop
    step = 0
    step_sample = -1 * CONFIG['sample_interval']
    train_iter = 1
    stru_in_iter = 0
    
    # Initialize Mahalanobis Calculator
    ma_calc = MahalanobisCalculator(
        dataset=trainset_atoms,
        species=CONFIG['species'],
        model_path=CONFIG['model_name'],
        device=CONFIG['device']
    )
    ma_calc.init_statistics()
    
    # Initial sampling
    write(CONFIG['trainset_file'], sample_structures(CONFIG['total_trainset_file'], CONFIG['k_sample_ratio']))

    try:
        while step < CONFIG['total_steps']:
            step += 1
            
            # Calculate Uncertainty
            dist = ma_calc.calculate_distance(atoms)
            is_uncertain = dist > CONFIG['threshold']
            print(f"Step {step}: Mahalanobis Dist = {dist:.4f}, Uncertain: {is_uncertain}")

            if is_uncertain:
                # --- DFT Calculation Path ---
                atoms.calc = get_dft_calculator()
                
                # Check if we should sample this structure
                if (step - step_sample) >= CONFIG['sample_interval']:
                    print(f"Step {step}: Sampling structure for labeling (DFT)...")
                    dyn.run(1)
                    
                    # Add to datasets
                    write(CONFIG['trainset_file'], atoms, append=True)
                    write(CONFIG['total_trainset_file'], atoms, append=True)
                    
                    stru_in_iter += 1
                    step_sample = step
                else:
                    # Skip sampling if too close to last sample
                    if train_iter > CONFIG['skip_iter']:
                        print(f'Step {step}: Enabling DFT labeling (Skip iter reached)')
                        dyn.run(1)
                    else:
                        print(f'Step {step}: Training phase, skipping labeling. Using MLFF.')
                        atoms.calc = SevenNetCalculator(model=CONFIG['model_name'], device=CONFIG['device'])
                        dyn.run(1)
            else:
                # --- MLFF Calculation Path ---
                print(f"Step {step}: Using MLFF (Normal)")
                atoms.calc = SevenNetCalculator(model=CONFIG['model_name'], device=CONFIG['device'])
                dyn.run(1)

            # --- Retraining Logic ---
            if stru_in_iter == CONFIG['num_stru_train']:
                print(f'\n>>> Starting Training Iteration {train_iter} (Step {step}) <<<')
                
                # Run SevenNet training
                # Note: working_dir is assumed to be current dir '.'
                train_v2(sevenn_config, '.') 
                
                print(f'>>> Training Iteration {train_iter} Finished <<<')

                # Update Uncertainty Calculator with new model and data
                new_dataset = read(CONFIG['total_trainset_file'], index=':')
                ma_calc = MahalanobisCalculator(
                    dataset=new_dataset,
                    species=CONFIG['species'],
                    model_path=CONFIG['model_name'] # Assumes model name stays same after save
                )
                ma_calc.init_statistics()
                print('Query algorithm parameters updated.')

                # Archive current trainset and prepare next batch
                os.rename(CONFIG['trainset_file'], f'trainset_iter_{train_iter}.extxyz')
                write(CONFIG['trainset_file'], sample_structures(CONFIG['total_trainset_file'], CONFIG['k_sample_ratio']))
                print('Dataset updated for next iteration.')

                # Cleanup temp data
                if os.path.exists('sevenn_data'):
                    shutil.rmtree('sevenn_data')

                train_iter += 1
                stru_in_iter = 0

    finally:
        csv_file.close()
        print("Simulation finished or terminated. CSV closed.")

if __name__ == "__main__":
    main()