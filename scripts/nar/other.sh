#!/bin/bash

for model_type in "GCN" "GatedGCN" "GAT"; do
    for num_neighbours in 4 8 16 32 64 80 96; do
        for hidden_dim in 64 128 256; do
            for k_hop in false true; do
                name_suffix=""
                if [ "$k_hop" = true ]; then
                    name_suffix="_k-hop"
                fi
                python main.py --repeat 3 --save_key_value_grad_metrics \
                    --cfg "configs/${model_type}/key-recall-${model_type}.yaml" \
                    name_tag "num_neighbours_${num_neighbours}_hidden_dim_${hidden_dim}${name_suffix}" \
                    key_recall.num_key_nodes "$num_neighbours" \
                    gnn.dim_inner "$hidden_dim" \
                    gnn.use_k_hop_aggregation "$k_hop"
            done
        done
    done
done