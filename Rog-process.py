import torch
import pickle
import networkx as nx
from collections import deque
import json
import argparse
from os.path import join as opj


def build_graph(sample):
    all_entities = sample["text_entity_list"] + sample["non_text_entity_list"]
    all_relations = sample["relation_list"]
    h_id_list, r_id_list, t_id_list = sample["h_id_list"], sample["r_id_list"], sample["t_id_list"]
    all_triplets = [(all_entities[h], all_relations[r], all_entities[t]) for (h,r,t) in zip(h_id_list, r_id_list, t_id_list)]
    
    G = nx.MultiDiGraph()
    for (a, b, c) in all_triplets:
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, relation=b)
    return G

def bfs_with_rule(graph, start_node, target_rule, max_p = 10):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                for key, edge_data in graph[current_node][neighbor].items():
                    rel = edge_data.get("relation")
                    if rel != target_rule[len(current_path)]:
                        continue
                    queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths

def bfs_with_rule_directional(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])  # (current_node, path)

    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            continue
        
        if current_node not in graph:
            continue

        next_rel = target_rule[len(current_path)]

        if not current_path or current_path[0][0] == start_node:
            for neighbor in graph.successors(current_node):
                for _, edge_data in graph[current_node][neighbor].items():
                    if edge_data.get("relation") == next_rel:
                        queue.append((neighbor, current_path + [(current_node, next_rel, neighbor)]))

        if not current_path or current_path[-1][-1] == start_node:
            for neighbor in graph.predecessors(current_node):
                for _, edge_data in graph[neighbor][current_node].items():
                    if edge_data.get("relation") == next_rel:
                        queue.append((neighbor, [(neighbor, next_rel, current_node)] + current_path))

    return result_paths

def remove_duplicate(results):
    seen = set()
    final = []
    for path in results:
        st_pair = (path[0][0], path[-1][-1])
        if st_pair not in seen:
            final.append(path)
            seen.add(st_pair)
    return final

def main():
    parser = argparse.ArgumentParser(description='CFRE')
    parser.add_argument('--dataset', type=str, default="webqsp", help='dataset used, option: ')
    parser.add_argument('--steps', type=str, default="960")
    parser.add_argument('--beam', type=int, default=5)

    args = parser.parse_args()

    dataset = pickle.load(open(f"datasets/{args.dataset}/processed/test.pkl", "rb"))
    read_dir = f'results/{args.dataset}/RoG-step-{args.steps}/test'
    with open(opj(read_dir, 'predictions_30_False.jsonl'), 'r', encoding='utf-8') as f:
        gen_relts = [json.loads(line) for line in f]

    paths_res = {}
    coverage = []
    path_sacle = []
    for data, relt in zip(dataset, gen_relts):
        relt['prediction'] = relt['prediction'][:args.beam]
        assert data['id'] == relt['id']
        answers = data['a_entity']
        graph = build_graph(data)
        results = []
        for entity in data['q_entity']:
            for rule in relt['prediction']:
                res = bfs_with_rule_directional(graph, entity, rule)
                results.extend(res)
        results = remove_duplicate(results)
        retrieved_ans = [a for a in answers if a in str(results)]
        if len(answers) > 0:
            coverage.append(len(retrieved_ans) / len(answers))
        path_sacle.append(len(results))
        paths_res[data['id']] = {'paths': results, 'q_entities': data['q_entity']}

    torch.save(paths_res, opj(read_dir, f'rog_ours-{args.beam}-bidir.pth'))
    import numpy as np

    print(args.dataset, args.steps, args.beam, np.mean(coverage), np.mean(path_sacle))


if __name__ == '__main__':
    main()