#include <iostream>
#include <algorithm>

using namespace std;
class grads{


	public:



///############## GRADIENT FOR LOSSES

//MAE LOSS
		float mae_grad(float y,float a){
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
		float bce_grad(float y, float a){
		a=clamp(a,1e-7f,1.0f-1e-7f);

		return (((1-y)/(1-a))-(y/a));

		}

};



int main(){


return 0;
}
