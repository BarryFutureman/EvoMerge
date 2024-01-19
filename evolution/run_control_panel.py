import time
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import json
import os


cool_blue = colors.Color(
    name="cool_blue",
    c50="#e3f2fd",
    c100="#bbdefb",
    c200="#90caf9",
    c300="#64b5f6",
    c400="#42a5f5",
    c500="#2196f3",
    c600="#1e88e5",
    c700="#1976d2",
    c800="#1565c0",
    c900="#0d47a1",
    c950="#07357f",
)


class Softy(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.indigo,
        secondary_hue: colors.Color | str = colors.rose,
    ):

        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
        )
        super().set(

        )


def define_population_tabs():
    if os.path.exists(dna_folder):
        gen_folders = []
        for filename in os.listdir(dna_folder):
            gen_folders.append(filename)
    else:
        raise NotImplementedError()
    print(gen_folders)
    with gr.Row():
        for index, g in enumerate(gen_folders):
            gen_folder = f"{dna_folder}/{g}"
            population_json_files = []
            for f in os.listdir(gen_folder):
                if f.endswith(".json"):
                    population_json_files.append(f"{gen_folder}/{f}")
            with gr.Tab(f"Gen{index:06d}"):
                define_gen_group(population_json_files)


def do_gen_refresh():
    global stop_refresh_gen_view_flag
    stop_refresh_gen_view_flag = False
    while True:
        if stop_refresh_gen_view_flag:
            break
        else:
            yield set_gen_view(-1)
        time.sleep(1)


def stop_refresh_gen():
    global stop_refresh_gen_view_flag
    stop_refresh_gen_view_flag = True


def check_dna_folder():
    if os.path.exists(dna_folder):
        gen_folders = []
        for filename in os.listdir(dna_folder):
            gen_folders.append(filename)
        # Yeah the naming of the folder is a bit awkward
        gen_folders = sorted(gen_folders, key=lambda x: int(x[3:]))
    else:
        raise NotImplementedError()
    return gen_folders


def set_gen_view(n=-1):
    global current_gen_index
    gen_folders = check_dna_folder()
    if n < 0:
        n = len(gen_folders) + n
    else:
        n = min(n, len(gen_folders)-1)
    current_gen_index = n
    new_json_nodes = define_population(gen_folders)

    return_nodes = [n]
    return_nodes.extend(new_json_nodes)
    return return_nodes


def define_population(gen_folders):
    json_nodes = []
    with gr.Row():
        g = gen_folders[current_gen_index]
        gen_folder = f"{dna_folder}/{g}"
        population_json_files = []
        for f in os.listdir(gen_folder):
            if f.endswith(".json"):
                population_json_files.append(f"{gen_folder}/{f}")
        with gr.Tab(f"Current Viewing"):
            json_nodes.extend(define_gen_group(population_json_files))

        index_last = max(0, current_gen_index - 1)
        g = gen_folders[index_last]
        gen_folder = f"{dna_folder}/{g}"
        population_json_files = []
        for f in os.listdir(gen_folder):
            if f.endswith(".json"):
                population_json_files.append(f"{gen_folder}/{f}")
        with gr.Tab(f"Previous Generation"):
            json_nodes.extend(define_gen_group(population_json_files))
    return json_nodes


def define_gen_group(population_json):
    json_nodes = []
    i = 0
    while i < len(population_json):
        with gr.Row():
            for j in range(i, min(i + 4, len(population_json))):
                with gr.Column(scale=1):
                    with open(population_json[j], 'r') as target_data_file:
                        data = json.load(target_data_file)
                        json_node = gr.Json(data)
                        json_nodes.append(json_node)
                i = j
        i += 1

    return json_nodes


def plot_fitness():
    gen_folders = check_dna_folder()
    print(gen_folders)
    mean_fitness_scores = []
    max_fitness_scores = []
    min_fitness_scores = []
    for gen_folder in gen_folders:
        gen_folder_path = f"{dna_folder}/{gen_folder}"
        group_fitness_scores = []
        for f in os.listdir(gen_folder_path):
            f = f"{gen_folder_path}/{f}"
            if f.endswith(".json"):
                with open(f, 'r') as target_data_file:
                    data = json.load(target_data_file)

                    group_fitness_scores.append(data["fitness"])


        mean_score = sum(group_fitness_scores) / len(group_fitness_scores)
        max_fitness_scores.append(max(group_fitness_scores))
        min_fitness_scores.append(min(group_fitness_scores))
        mean_fitness_scores.append(mean_score)

    # Plot the mean scores
    gen_labels = [g for g in range(len(gen_folders))]
    """plt.plot(gen_labels, mean_fitness_scores, color='blue')
    plt.grid = True
    plt.xlabel('Generations')
    plt.ylabel('Mean Scores')
    plt.title('Generation Mean Scores')"""
    line_labels = ["Mean" for _ in mean_fitness_scores] + \
                  ["Max" for _ in max_fitness_scores] + \
                  ["Min" for _ in min_fitness_scores]
    score_points = mean_fitness_scores + max_fitness_scores + min_fitness_scores
    data = {"Lines": line_labels, 'Generations': gen_labels*3, 'Score': score_points}
    pd_df = pd.DataFrame(data)

    return gr.LinePlot(
        value=pd_df,
        x="Generations",
        y="Score",
        title="Fitness",
        color="Lines",
        color_legend_position="bottom",
        width=1024,
        height=256,
    )


simulation_folder = "C:/Files/PycharmProjects/NeuralEvolution/EvoMerge/runs/evolve_tiny_llamas"
dna_folder = simulation_folder + "/DNAs"
current_gen_index = 0
stop_refresh_gen_view_flag = False

with gr.Blocks(title="Dataset Viewer", theme=Softy()) as demo:
    # Guess tab
    with gr.Tab("View Population") as population_view:
        with gr.Row():
            gen_json_nodes = define_population(check_dna_folder())
        with gr.Row():
            with gr.Column(scale=1):
                gen_view_textbox = gr.Number(label="Generation")
                textbox_output_update_group = [gen_view_textbox]
                textbox_output_update_group.extend(gen_json_nodes)
                gen_view_textbox.submit(fn=set_gen_view, inputs=gen_view_textbox,
                                        outputs=textbox_output_update_group)
            with gr.Column(scale=1):
                refresh_gen_viewer_button = gr.Button(value="Refresh", variant="primary")
                refresh_gen_viewer_button.click(fn=do_gen_refresh, inputs=None,
                                                outputs=textbox_output_update_group)
                stop_refresh_gen_viewer_button = gr.Button(value="Stop Refresh", variant="secondary")
                stop_refresh_gen_viewer_button.click(fn=stop_refresh_gen, inputs=None, outputs=None)

    with gr.Tab("Graph"):
        fitness_plt = plot_fitness()
        with gr.Row():
            refresh_fitness_plt_button = gr.Button(value="Refresh", variant="primary")
            refresh_fitness_plt_button.click(fn=plot_fitness, inputs=None, outputs=fitness_plt, every=30)

    with gr.Tab("Settings"):
        pass


demo.queue()
demo.launch(max_threads=8, server_port=5000, share=False)
