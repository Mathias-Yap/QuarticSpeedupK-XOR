import numpy as np
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.problem_set_generation.planted_noisy_kxor_generator import PlantedNoisyKXORGenerator
import shutil

class KXORDatasetGenerator:
    """
    Generator for K-XOR problem instance datasets over a grid of parameters.
    """
    def generate_kxor_dataset_grid(
        self,
        folder_path: str,
        n_min: int = 3,
        n_max: int = 10,
        k_min: int = 2,
        k_max: int = 5,
        rho_step: float = 0.2,
        m_constant: float = 0.25,
        seed: int = 123,
        zip: bool = False
    ) -> None:
        """
        - n in [n_min, n_max]
        - k in [k_min, k_max]
        - rho in {0, rho_step, 2*rho_step, ..., 1}
        - m = ceil(m_constant * n * log n)  (Θ(n log n))

        For every (n, k) combination:
        - one random instance (R_{n,k}(m))
        - for every rho, one planted instance (P^z_{n,k}(m, ρ))
        """
        rng = np.random.default_rng(seed)
        instance_id = 0

        # noise grid (ρ values)
        num_steps = int(1.0 / rho_step)
        rho_values = [round(i * rho_step, 10) for i in range(num_steps + 1)]  # include 1.0
        
        # loop over all parameter combinations
        for n in range(n_min, n_max + 1):
            for k in range(k_min, k_max + 1):
                if k > n:
                    continue  # can't have clause of size k > n

                # m = Θ(n log n)
                m = int(np.ceil(m_constant * n * np.log(n)))

                gen = PlantedNoisyKXORGenerator(n=n, k=k, rng=rng)

                # random instance
                random_instance = gen.sample_random(m=m)
                random_instance.save(f"{folder_path}/kxor_instance_id{instance_id}_n{n}_k{k}_m{m}_rho{0}.npz")
                instance_id += 1
                
                # planted instances for each ρ
                for rho in rho_values:
                    planted_instance = gen.sample_planted(m=m, rho=rho)
                    planted_instance.save(f"{folder_path}/kxor_instance_id{instance_id}_n{n}_k{k}_m{m}_rho{rho}.npz")
                    instance_id += 1

        if zip:
            shutil.make_archive(folder_path, 'zip', folder_path)
    
    def generate_kxor_dataset_explicit_params(
        self,
        folder_path: str,
        params: list[tuple[int,int,int,float]],
        seed: int = 123,
        zip: bool = False
    ) -> None:
        """Generate K-XOR problem instance datasets for explicit parameter sets.
        Parameters:
            folder_path (str): Path to save the generated instances.
            params (list of tuples): Each tuple is (n, k, m, rho).
                    - n (int): Number of variables.
                    - k (int): Clause size.
                    - m (int): Number of clauses.
                    - rho (float): Planted advantage (0.0 for random instance).
            seed (int): Random seed for reproducibility.
            zip (bool): Whether to zip the output folder.
        """
        rng = np.random.default_rng(seed)
        instance_id = 0

        for (n, k, m, rho) in params:
            gen = PlantedNoisyKXORGenerator(n=n, k=k, rng=rng)

            if rho == 0.0:
                # random instance
                random_instance = gen.sample_random(m=m)
                random_instance.save(f"{folder_path}/kxor_instance_n{n}_k{k}_m{m}_rho{rho}.npz")
                instance_id += 1
            else:
                # planted instance
                planted_instance = gen.sample_planted(m=m, rho=rho)
                planted_instance.save(f"{folder_path}/kxor_instance_n{n}_k{k}_m{m}_rho{rho}.npz")
                instance_id += 1

        if zip:
            shutil.make_archive(folder_path, 'zip', folder_path)

    def generate_single_instance(
        self,
        folder_path: str,
        n: int,
        k: int,
        m: int,
        rho: float,
        seed: int = 123
    ) -> str:
        """Generate a single K-XOR problem instance with specified parameters.
        Parameters:
            folder_path (str): Path to save the generated instance.
            n (int): Number of variables.
            k (int): Clause size.
            m (int): Number of clauses.
            rho (float): Planted advantage (0.0 for random instance).
            seed (int): Random seed for reproducibility.
        Returns:
            str: Path to the saved instance file.
        """
        rng = np.random.default_rng(seed)
        gen = PlantedNoisyKXORGenerator(n=n, k=k, rng=rng)

        if rho == 0.0:
            # random instance
            instance = gen.sample_random(m=m)
        else:
            # planted instance
            instance = gen.sample_planted(m=m, rho=rho)

        instance.save(f"{folder_path}/kxor_instance_n{n}_k{k}_m{m}_rho{rho}.npz")
        
        return f"{folder_path}/kxor_instance_n{n}_k{k}_m{m}_rho{rho}.npz"

