/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include "optimization.h"

// Test the energy minimzation

class ExampleEnergy: public EnergyFunction {
public:
	virtual VectorXf initialValue() {
		return VectorXf::Zero( 2 );
	}
	virtual double gradient( const VectorXf & x, VectorXf & dx ) {
		double fx = (x[0] - 1)*(x[0] - 6) + (x[0] - 4)*(x[1] - 2)*(x[0] - 4)*(x[1] - 2) + x[1]*x[1];
		dx[0] = 2*x[0] - 7 + 2*(x[1] - 2)*(x[0] - 4)*(x[1] - 2);
		dx[1] = 2*x[1] + 2*(x[0] - 4)*(x[1] - 2)*(x[0] - 4);
		return fx;
	}
};


int main() {
	ExampleEnergy e;
	VectorXf m = minimizeLBFGS( e, true );
	
	VectorXf g( m.rows() );
	std::cout<<"Minimized "<<e.gradient(m,g)<<std::endl;
	std::cout<<g<<std::endl;

}
