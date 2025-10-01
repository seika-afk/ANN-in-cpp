#include "Gradients.h"
#include <algorithm>
#include <cmath>

using namespace std;



///############## GRADIENT FOR LOSSES

//MAE LOSS
		float Gradients::mae_grad(float y,float a){
			// FOR WHY IS IT USED : CHECK MY NOTES (if u can understand it)
			if (y>a){
				return -1.0f;
			}
			else if (y<a){

			return 1.0f;
			}
			else{

			return 0.0f;}
		}


//BCE LOSS
		float Gradients::bce_grad(float y, float a){
		a=clamp(a,1e-7f,1.0f-1e-7f);

		return (((1-y)/(1-a))-(y/a));

		}





//######################## Gradients for Activation Functions
	
//#### RELU
		float Gradients::relu_grad(float x){

		return (x<= 0? 0.0f :1.0f );

	}
//#### SIGMOID :
		float Gradients::sigmoid_grad(float x){

		return exp(-x)/pow((1+(exp(-x))),2);



		}





