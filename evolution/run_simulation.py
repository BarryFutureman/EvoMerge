import random
import time
import gc
import torch
from crossover_methods import slerp_merge_crossover
from eval_methods import random_lm_eval
from mutation_methods import short_qlora_mutation
from selection_methods import roulette_wheel_based_selection
from population_scheduler import PopulationScheduler
from evo_lm import EvoLM
import os


def run(simulation_folder):
    generation_index = 0
    dna_folder = simulation_folder + "/DNAs"
    if os.path.exists(dna_folder):
        existing_iterations = []
        for filename in os.listdir(dna_folder):
            existing_iterations.append(int(filename[3:]))
        if existing_iterations:
            generation_index = max(existing_iterations)
    else:
        os.mkdir(dna_folder)

    population = []
    population_folder = simulation_folder + "/population"
    if os.path.exists(population_folder) and os.path.isdir(population_folder):
        # Loop through the files in the folder
        for filename in os.listdir(population_folder):
            # Get the absolute path of the file
            file_path = f"{population_folder}/{filename}"
            if not os.path.isdir(file_path):
                continue

            lm = EvoLM(model_folder=file_path, dna_id=filename)
            population.append(lm)

    # configuration
    population_size = len(population)
    target_iterations = 12
    keep_top_k = 4
    mutate_count = 2

    # configurate population_scheduler
    min_population_size = population_size
    max_population_size = 12
    expand_steps = 2
    population_scheduler = PopulationScheduler(population_size, max_population_size=max_population_size,
                                               min_population_size=min_population_size,
                                               total_generations=target_iterations, expand_steps=expand_steps)

    print("Population Curve:", [population_scheduler.get_population_size(i) for i in range(target_iterations)])
    while not input("Type 'C' to continue: ").lower() == "c":
        pass

    # Run simulation loop
    evaluation_process = random_lm_eval.RandomEval(dataset_dir=simulation_folder+"/datasets",
                                                   sample_size=20)
    selection_process = roulette_wheel_based_selection.RouletteWheelSelection(keep_top_k=keep_top_k)
    crossover_process = slerp_merge_crossover.SlerpMergeCrossover(population_folder=population_folder)
    mutation_process = short_qlora_mutation.ShortQloraMutation(keep_top_k=keep_top_k,
                                                               target_mutation_count=mutate_count,
                                                               mutation_strength=16)
    for iteration in range(target_iterations):
        # Sanity check: clear cuda cache
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: Issue with more than 8 items being created is still not fixed,
        #  debug it by letting the seed set in sft mutation
        population = evaluation_process.run_evaluation(population)
        print("=====================================================")
        for j, p in enumerate(population):
            print(p, ":", population[j].DNA)
            # Save the population DNAs here
            curr_gen_folder = dna_folder + f"/Gen{generation_index + iteration}"
            if not os.path.exists(curr_gen_folder):
                os.mkdir(curr_gen_folder)
            p.DNA.save_as_json(curr_gen_folder)

        if iteration == target_iterations - 1:
            # End the final iteration here
            break

        # print("Take a break")
        # time.sleep(20)

        next_population_size = population_scheduler.get_population_size(iteration)
        selected_pairs, new_population = selection_process.run_selection(population, next_population_size)
        print("selected_pairs:", selected_pairs)

        new_population = crossover_process.run_crossover(old_population=population,
                                                         new_population=new_population,
                                                         pairs=selected_pairs)
        # Sanity check: clear cuda cache
        gc.collect()
        torch.cuda.empty_cache()

        # Delete old population
        for lm in population:
            # We don't want to delete candidates in the new population
            if lm not in new_population:
                lm.delete_files()
        population = new_population

        population = mutation_process.run_mutation(population)


if __name__ == '__main__':
    run("C:/Files/PycharmProjects/NeuralEvolution/EvoMerge/runs/evolve_tiny_llamas")







