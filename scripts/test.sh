echo "Testing $2 - $3 - $4 - $5"
./darknet detector map $2/obj.data $3 $1 -iou_thresh $5 > $1_$4.txt 2>&1