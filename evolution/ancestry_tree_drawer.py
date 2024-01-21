import os
import json
import graphviz


def draw_full_tree(simulation_folder):
    dna_folder = simulation_folder + "/DNAs"
    gen_folders = []
    if os.path.exists(dna_folder):
        gen_folders = []
        for filename in os.listdir(dna_folder):
            gen_folders.append(filename)
        # Yeah the naming of the folder is a bit awkward
        gen_folders = sorted(gen_folders, key=lambda x: int(x[3:]))
    else:
        raise NotImplementedError()

    population_json_data = []
    for generation_index, gen_folder_name in enumerate(gen_folders):
        gen_folder = f"{dna_folder}/{gen_folder_name}"
        generation_group = []
        for f in os.listdir(gen_folder):
            if f.endswith(".json"):
                p_json_file = f"{gen_folder}/{f}"
                with open(p_json_file, 'r') as target_data_file:
                    target_data = json.load(target_data_file)
                generation_group.append(target_data)

        population_json_data.append(generation_group)

    for i, p in enumerate(population_json_data):
        population_json_data[i] = p[:]
    population_json_data = population_json_data[:]

    def get_node_id(individual_id, curr_gen):
        return individual_id + f"-gen{curr_gen}"

    def string_to_hex(input_str):
        # Take the first three characters
        c1, c2, c3 = input_str[:3]

        # Clamp each character to be within the valid hexadecimal range
        c1 = min(max(c1, '0'), 'F').upper()
        c2 = min(max(c2, '0'), 'F').upper()
        c3 = min(max(c3, '0'), 'F').upper()

        # Construct the hex string
        hex_string = f"#6{c1}6{c2}6{c3}"

        return hex_string

    def draw_family_tree(population_data):
        dot = graphviz.Digraph(format='png')
        graph_attr = {
            'bgcolor': '#3D3B40',
            "ranksep": '2',
            'dpi': '128',
        }
        dot.attr(**graph_attr)

        previous_gen_ids = []
        curr_gen_ids = []
        for generation, individuals in enumerate(population_data):
            s = graphviz.Digraph(f'gen{generation}')
            previous_gen_ids = curr_gen_ids.copy()
            curr_gen_ids.clear()
            for individual in individuals:
                curr_gen_ids.append(individual["id"])
                node_id = get_node_id(individual["id"], generation)

                mutation_info = "null"
                if individual['mutation_info']:
                    mutation_info = str(individual['mutation_info']['datasets'])

                node_label = f"{node_id} <{individual['model_name']}>\n" \
                             f"Fitness: {individual['fitness']:.2f}\n" \
                             f"Mutation: {mutation_info}"
                node_attr = {
                    'shape': 'hexagon' if individual["mutation_info"] is not None else "invhouse",
                    'color': f"{string_to_hex(node_id)}",  # '#365486',
                    'style': 'filled',
                    'fontname': 'arial',
                    'peripheries': '4' if individual["mutation_info"] is not None else "0",
                    'fontcolor': '#FFFBF5',
                    'fontsize': '10',
                    'xlabel': f"{individual['fitness']:.2f}",
                    'fontstyle': 'italic',
                    # 'fillcolor': 'lightgray',
                }
                s.node(node_id, label=node_label, **node_attr)

                if generation > 0:
                    # We check whether it was not a crossover product
                    if individual["id"] in previous_gen_ids:
                        individual['parents'] = [individual["id"]]
                    elif individual['parents'] is None:
                        continue

                    p_lst = individual['parents']
                    for p_index, parent_id in enumerate(p_lst):
                        parent_id = get_node_id(parent_id, generation-1)
                        if len(p_lst) < 2:
                            edge_attr = {
                                'color': 'gray',
                                'style': 'dashed',
                                'fontsize': '8',
                            }
                        else:
                            colors = ["#525CEB", "#F87474"]
                            edge_attr = {
                                'color': colors[p_index],
                                'style': 'solid',
                                'fontsize': '8',
                            }
                        s.edge(parent_id, node_id, **edge_attr)

            dot.subgraph(s)

        file_name = "tree"
        dot.render(file_name, cleanup=True, format='png', view=False)
        print(f"Family tree has been generated as '{file_name}.png'.")
        return f'{file_name}.png'

    file = draw_family_tree(population_json_data)

    return file
