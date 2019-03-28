#!/bin/bash
for i in 1 2; do
	hyst -t hycreate '' -i test${i}.xml -o test${i}-converted.hyc2
done

