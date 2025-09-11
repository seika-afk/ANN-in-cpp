#include <iostream>
#include <vector>

using namespace std;
class ANN{

	public :
	string ac_fn;

	
	//ANN(int neurons, int h_layers,string ac_fn){
		//this->ac_fn=ac_fn;
	//}

	void forward_pass(vector<float> inputs){
		for(int i =0; i<inputs.size();i++){
			cout<<"Input "<<i<<" : "<<inputs[i]<<"\n";
			
		}}

	};

int main(){

vector<float> inputs ={1,2,3};

ANN ann;
ann.forward_pass(inputs);
return 0;
}





