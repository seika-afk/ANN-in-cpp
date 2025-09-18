#include "Losses.h"
#include <cmath>

using namespace std;

	//###################### MEAN ABSOLUTE ERROR
	float Losses::mae(float yi,float xi){

	float error =abs(yi-xi);

return error;
	}


	//######################### LOG LOSS / BINARY CROSS ENTROPY
	

	float Losses::bce(float yi,float xi){
    if (xi <= 0.0f) xi = 1e-7f;
    if (xi >= 1.0f) xi = 1.0f - 1e-7f;

	float error =-((yi*log(xi))+(1-yi)*(log(1-xi)));

return error;
	}





