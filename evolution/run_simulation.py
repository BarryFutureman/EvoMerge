import simulation_configs
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
    configs_folder = simulation_folder + "/configs"
    simulation_config = simulation_configs.load_config(configs_folder + "/example_run.yaml")
    print(simulation_config)

    population_size = len(population)
    min_population_size = population_size
    max_population_size = simulation_config.max_population_size
    target_iterations = simulation_config.target_iterations

    # configurate population_scheduler
    population_scheduler = PopulationScheduler(population_size, max_population_size=max_population_size,
                                               min_population_size=min_population_size,
                                               total_generations=simulation_config.target_iterations,
                                               expand_steps=simulation_config.expand_steps)

    print("Population Curve:", [population_scheduler.get_population_size(i) for i in range(target_iterations)])
    while not input("Type 'C' to continue: ").lower() == "c":
        pass

    # Run simulation loop
    evaluation_process = random_lm_eval.RandomEval(dataset_dir=simulation_folder+"/datasets",
                                                   config=simulation_config)
    selection_process = roulette_wheel_based_selection.RouletteWheelSelection(config=simulation_config)
    crossover_process = slerp_merge_crossover.SlerpMergeCrossover(population_folder=population_folder,
                                                                  config=simulation_config)
    mutation_process = short_qlora_mutation.ShortQloraMutation(config=simulation_config)
    for iteration in range(target_iterations):
        # Sanity check: clear cuda cache
        gc.collect()
        torch.cuda.empty_cache()

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
    run("C:/Files/PycharmProjects/EvoMerge/EvoMerge/runs/quick_evolve")







