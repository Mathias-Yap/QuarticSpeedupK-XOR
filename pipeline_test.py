import os
from typing import Sequence
from kxor_code.algorithms.base_alg_step import NoOpStep, ProblemRecord, StepConfig, StepStats, StepContext, BaseAlgorithmStep
from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.algorithms.power_iteration_step import PowerIterationStep
from kxor_code.algorithms.quartic_quantum_algorithm_step import QuarticQuantumAlgorithm
from kxor_code.algorithms.threshold_step import ThresholdStep
from kxor_code.problem_set_generation.kxor_instance import KXORInstance
from kxor_code.algorithms.algorithm_pipeline import AlgorithmPipeline
from kxor_code.problem_set_generation.kxor_dataset_generator import KXORDatasetGenerator
import pickle

step = ComputeKikuchiStep()
step2 = ThresholdStep()
folder_path = "data/kxor_dataset"
generator = KXORDatasetGenerator()
step3 = ClassicalEigenvaluesStep()
step3.time_limit = 600  # Set time limit to 600 seconds
pipeline = AlgorithmPipeline(steps=[step, step2, step3], verbose=True)
generator = KXORDatasetGenerator()

params_list = pickle.load(open("theorem_results.pkl", "rb"))
for params in params_list:
    n = params["n"]
    k = params["k"]
    m = params["m"]
    kappa = params["kappa"]
    ell = params["ell"]
    rho = 0.8
    failure_prob = params["failure_probability_bound"]
    instance_path = generator.generate_single_instance(
        folder_path=folder_path,
        n=n,
        k=k,
        m=m,
        rho=rho,
        seed = 42,
    )
    instance = KXORInstance.load(instance_path)
    problem = ProblemRecord(problem_id="test_problem", instance=instance)
    problem.add_field("ell", ell)
    problem.add_field("kappa", kappa)
    problem.add_field("failure_prob", failure_prob)
    problem.add_field("rho", rho)
    pipeline.run(problem)
    problem.save_compact(f"data/results/problem_n{n}_k{k}_m{m}_rho{rho}_kappa{kappa}_ell{ell}.pkl")