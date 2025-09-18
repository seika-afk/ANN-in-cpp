#include <iostream>
#include <cmath>

using namespace std;




class Losses{
	
	public:


	//###################### MEAN ABSOLUTE ERROR
	float mae(float yi,float xi){

	float error =abs(yi-xi);

return error;
	}


	//######################### LOG LOSS / BINARY CROSS ENTROPY
	

	float bce(float yi,float xi){

	float error =-((yi*log(xi))+(1-yi)*(log(1-xi)));

return error;
	}





};

int main(){

float xi=0.45;
float yi=0.6;

Losses loss;
// sequence : real output,predicted output
cout<<"Mean Absolute error : "<<loss.mae(yi,xi)<<endl;
cout <<"Binary Cross Entropy : "<<loss.bce(yi,xi)<<endl;



return 0;
}
