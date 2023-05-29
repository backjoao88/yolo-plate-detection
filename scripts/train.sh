echo "[INFO] Treinando o modelo $1 com o dataset $2 (./darknet detector train $2/obj.data $1 -dont_show -ext_output | tee $1.txt)..." 
./darknet detector train $2/obj.data $1 -dont_show -ext_output | tee $1.txt

# --------------------
# !./darknet detector map ./../data_ccpd_weather/obj.data ./../models/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224.cfg ./../data_ccpd_weather/backup/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224_2000.weights -iou_thresh 0.3
# !./darknet classify ./../models/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224.cfg ./../data_ccpd_weather/backup/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224_2000.weights './../data_ccpd_weather/obj/0119-0_0-246&593_409&654-409&652_247&654_246&595_408&593-0_0_25_33_27_28_33-99-39.jpg'
# !./darknet detect ./../models/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224.cfg ./../data_ccpd_weather/backup/yolov3-tiny-2000-32-1600-1800-1-18-adam-0001-09-224-224_2000.weights './../data_ccpd_weather/obj/0185-3_0-297&500_491&580-491&567_301&580_297&513_487&500-11_1_13_33_8_27_31-131-48.jpg'
