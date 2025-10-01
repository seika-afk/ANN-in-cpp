#ifndef GRADIENTS_H
#define GRADIENTS_H


class Gradients
{
	public:
	float mae_grad(float y,float a);
	float bce_grad(float y, float a);
	float relu_grad(float x);
	float sigmoid_grad(float x);



};
#endif





