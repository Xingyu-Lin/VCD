# Notes on pre-trained model
Our pre-trained model, pre-collected dataset and the cached initial states for planning can be accessed throuth this google drive link: [Google Drive Link](https://drive.google.com/drive/folders/1gS8ejcY1imKVT8TD8zmNC38gNicpkL6X?usp=sharing)

The goolge drive folder includes:
* dataset: Pre-collected dataset for training.
* dynamics_model:  
  - `vsbl_dyn_best.pth`: Pre-trained dynamics GNN for partially observed point cloud.
  - `best_state.json`: Loading the pretraine edge and dynamics GNN will require a corresponding `best_state.json` that stores the model information. Just put these json files under the same directory as the pre-trained model. 
* edge_model:  
  - `vsbl_edge_best.pth`: Pre-trained Edge GNN.
  - `best_state.json`: Loading the pretraine edge and dynamics GNN will require a corresponding `best_state.json` that stores the model information. Just put these json files under the same directory as the pre-trained model. 
* cached_states:   
  - `1213_release_n1000.pkl`: The cached initial states for generating the training data.
  - `cloth_flatten_init_states_test_40.pkl` and `cloth_flatten_init_states_test_40_2.pkl`: in total 40 initial states for testing cloth smoothing on square clothes (each cached file has 20 states).
  - `cloth_flatten_test_retangular_1.pkl` and `cloth_flatten_test_retangular_1.pkl`: Initial states for testing on rectangular clothes (each cached file has 20 states).
  - `tshirt_flatten_init_states_small_2021_05_28_01_22.pkl` and `tshirt_flatten_init_states_small_2021_05_28_01_16.pkl`: Initial states for testing on t-shirt (each cahced file has 20 states).
