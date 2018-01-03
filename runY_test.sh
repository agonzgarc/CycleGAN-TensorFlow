#!/bin/bash

for i in {1..8}
do
   python inference.py --input inputs/Y/input$i.png --output outputs/Y/output$i.jpg 
done
