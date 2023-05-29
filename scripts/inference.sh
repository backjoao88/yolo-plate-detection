echo "[INFO] Inferindo modelo..."
./darknet detector demo $2/obj.data $3 $1 -i 0 -ext_output | tee inference-.txt

