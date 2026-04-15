:: TRAINING
:: python ...\landformAI\train.py --config ...runs\unetpp_enet_b7_rgb_full.yaml

:: TESTING
:: select the epoch based on the loss function / quality function plots

:: python ...\landformAI\test.py^
	:: --config...runs\unetpp_enet_b7_rgb_full.yaml
	:: --epoch 16^
	:: --probs

:: APPLY
::python  ...\landformAI\apply.py^
::	--config ...runs\unetpp_enet_b7_rgb_full.yaml^
::	--epoch 16^
::	--chunk-size 256^
::	--stride 128^
::	--input ...\data\all_tiles\dtm^
::	--odir ...\data\all_tiles\dtm_pred_enet_b7^
::	--export-prob
	
