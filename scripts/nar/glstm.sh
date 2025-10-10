#!/bin/bash

for num_neighbours in 4 8 16 32 64 80 96; do
    for memory_dim in 8 16 32; do
        for k_hop in false true; do
            name_suffix=""
            if [ "$k_hop" = true ]; then
                name_suffix="_k-hop"
            fi
            python main.py --repeat 3 --save_key_value_grad_metrics \
                --cfg "configs/GNN-xLSTM/key-recall-GNN-xLSTM.yaml" \
                name_tag "num_neighbours_${num_neighbours}_memory_dim_${memory_dim}${name_suffix}" \
                key_recall.num_key_nodes "$num_neighbours" \
                gnn.dim_inner "$((2 * memory_dim))" \
                xlstm.memory_dim "$memory_dim" \
                gnn.use_k_hop_aggregation "$k_hop"
        done
    done
done