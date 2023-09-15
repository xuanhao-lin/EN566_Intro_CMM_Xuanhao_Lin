.PHONY: plot write read clean

FXN := sin,cos,sinc
TXT := filename.txt
FMT := jpeg,eps,pdf

all: plot write read

default:all

plot:
	python trigonometry.py --function=${FXN} --print=${FMT}

write:
	python trigonometry.py --function=${FXN} --write=${TXT}

read:
	python trigonometry.py --function=${FXN} --read_from_file=${TXT}

clean:
	rm -f *.txt Plot_of_Trigonometric_Functions.*
