.PHONY: plot write read

FXN := sin,cos,sinc
TXT := filename.txt
FMT := jpeg,eps,pdf

plot:
	python trigonometry.py --function=${FXN} --print=${FMT}

write:
	python trigonometry.py --function=${FXN} --write=${TXT}

read:
	python trigonometry.py --read_from_file=${TXT}

