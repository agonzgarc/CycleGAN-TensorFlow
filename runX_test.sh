#!/bin/bash

for i in {1..8}
do
   python inference.py --input inputs/X/input$i.png --output outputs/output$i.jpg 
done
