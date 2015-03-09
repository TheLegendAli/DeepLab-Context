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
#include "permutohedral.h"
#include "optimization.h"
#include <iostream>

// This class simply tests the gradient computation on the permutohedral lattice

class PermutohedralEnergy: public EnergyFunction {
protected:
  Permutohedral p;
  MatrixXf f_, v0_, v1_;
public:
  PermutohedralEnergy( const MatrixXf & f, const MatrixXf & v0, const MatrixXf & v1 ):f_(f),v0_(v0), v1_(v1) {
    p.init( f );
  }
  virtual VectorXf initialValue() {
    MatrixXf f = f_;
    f.resize( f_.cols()*f_.rows(), 1 );
    return f;
  }
  virtual void setInitialValue( const VectorXf & v ) {
    MatrixXf f = v;
    f.resize( f_.rows(), f_.cols() );
    f_ = f;
  }
  virtual double gradient( const VectorXf & x, VectorXf & dx ) {
    MatrixXf f = x;
    f.resizeLike( f_ );
    p.init( f );
		
    MatrixXf bv = p.compute( v1_, true );
    dx = 0*x;
    p.gradient( dx.data(), v0_.data(), v1_.data(), v0_.rows() );
		
    return (bv.array()*v0_.array()).sum();
  }
};

int main() {
  int N = 1000, M = 4, d = 4;
  MatrixXf f = 0.3*MatrixXf::Random( d, N );
  MatrixXf a = MatrixXf::Random( M, N ), b = MatrixXf::Random( M, N );
  MatrixXf p;
  PermutohedralEnergy e( f.array(), a, b );
  if( !p.cols() || !p.rows() )
    p = e.initialValue();
  else {
    p.array() += 0.1;
    e.setInitialValue( p );
  }
  VectorXf g = p;
  std::cout<<"start = "<<e.gradient( p, g )<<std::endl;
  std::cout<<p.transpose()<<std::endl;
  p = minimizeLBFGS( e, 5, 0 );
  std::cout<<"E =  "<<e.gradient( p, g )<<"   "<<gradCheck( e, p, 1e-2 )<<std::endl;
  std::cout<<"g  = "<<g.transpose()<<std::endl;
  std::cout<<"ng = "<<numericGradient( e, p ).transpose()<<std::endl;
  std::cout<<"ng = "<<numericGradient( e, p, 1e-2 ).transpose()<<std::endl<<std::endl;
  int id;
  VectorXf dg = g-numericGradient( e, p );
  dg.array().abs().maxCoeff( &id );
  std::cout<<computeFunction( e, p-g, 0.02*g ).transpose()<<std::endl;
}
