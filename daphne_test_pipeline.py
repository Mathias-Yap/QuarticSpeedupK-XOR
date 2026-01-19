from kxor_code.algorithms.algorithm_pipeline import AlgorithmPipeline
from kxor_code.algorithms.base_alg_step import ProblemRecord
from kxor_code.algorithms.classical_eigenvalues_step import ClassicalEigenvaluesStep
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.algorithms.threshold_step import ThresholdStep
import os

from kxor_code.problem_set_generation.kxor_instance import KXORInstance



folder_path = "z_Daphne/data/problem_instances"
step = ComputeKikuchiStep()
step2 = ThresholdStep()
folder_path = "data/kxor_dataset"
step3 = ClassicalEigenvaluesStep()
pipeline = AlgorithmPipeline(steps=[step, step2, step3], verbose=True)
file_paths =  os.listdir(folder_path)
for file_name in file_paths:
    if file_name.endswith(".npz"):
        instance_path = os.path.join(folder_path, file_name)
        instance = KXORInstance.load(instance_path)
        problem =  ProblemRecord(problem_id="test_problem", instance=instance)
        pipeline.run(problem)
        problem.save_compact(f"data/results/{file_name.replace('.npz', '_result.pkl')}")
        break