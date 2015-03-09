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

#include "densecrf.h"
#include "optimization.h"
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "ppm.h"
#include "common.h"

// The energy object implements an energy function that is minimized using LBFGS
class CRFEnergy: public EnergyFunction {
protected:
  VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
  DenseCRF & crf_;
  const ObjectiveFunction & objective_;
  int NIT_;
  bool unary_, pairwise_, kernel_;
  float l2_norm_;
public:
  CRFEnergy( DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crf_(crf),objective_(objective),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f){
    initial_u_param_ = crf_.unaryParameters();
    initial_lbl_param_ = crf_.labelCompatibilityParameters();
    initial_knl_param_ = crf_.kernelParameters();
  }
  void setL2Norm( float norm ) {
    l2_norm_ = norm;
  }
  virtual VectorXf initialValue() {
    VectorXf p( unary_*initial_u_param_.rows() + pairwise_*initial_lbl_param_.rows() + kernel_*initial_knl_param_.rows() );
    p << (unary_?initial_u_param_:VectorXf()), (pairwise_?initial_lbl_param_:VectorXf()), (kernel_?initial_knl_param_:VectorXf());
    return p;
  }
  virtual double gradient( const VectorXf & x, VectorXf & dx ) {
    int p = 0;
    if (unary_) {
      crf_.setUnaryParameters( x.segment( p, initial_u_param_.rows() ) );
      p += initial_u_param_.rows();
    }
    if (pairwise_) {
      crf_.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
      p += initial_lbl_param_.rows();
    }
    if (kernel_)
      crf_.setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );
		
    VectorXf du = 0*initial_u_param_, dl = 0*initial_u_param_, dk = 0*initial_knl_param_;
    double r = crf_.gradient( NIT_, objective_, unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );
    dx.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
    dx << -(unary_?du:VectorXf()), -(pairwise_?dl:VectorXf()), -(kernel_?dk:VectorXf());
    r = -r;
    if( l2_norm_ > 0 ) {
      dx += l2_norm_ * x;
      r += 0.5*l2_norm_ * (x.dot(x));
    }
		
    return r;
  }
};

int main( int argc, char* argv[]){
  if (argc<4){
    printf("Usage: %s image annotations output\n", argv[0] );
    return 1;
  }
  // Number of labels [use only 4 to make our lives a bit easier]
  const int M = 4;
  // Load the color image and some crude annotations (which are used in a simple classifier)
  int W, H, GW, GH;
  unsigned char * im = readPPM( argv[1], W, H );
  if (!im){
    printf("Failed to load image!\n");
    return 1;
  }
  unsigned char * anno = readPPM( argv[2], GW, GH );
  if (!anno){
    printf("Failed to load annotations!\n");
    return 1;
  }
  if (W!=GW || H!=GH){
    printf("Annotation size doesn't match image!\n");
    return 1;
  }
  // Get the labeling
  VectorXs labeling = getLabeling( anno, W*H, M );
  const int N = W*H;
	
  // Get the logistic features (unary term)
  // Here we just use the color as a feature
  MatrixXf logistic_feature( 4, N ), logistic_transform( M, 4 );
  logistic_feature.fill( 1.f );
  for( int i=0; i<N; i++ )
    for( int k=0; k<3; k++ )
      logistic_feature(k,i) = im[3*i+k] / 255.;
	
  for( int j=0; j<logistic_transform.cols(); j++ )
    for( int i=0; i<logistic_transform.rows(); i++ )
      logistic_transform(i,j) = 0.01*(1-2.*random()/RAND_MAX);
	
  // Setup the CRF model
  DenseCRF2D crf(W, H, M);
  // Add a logistic unary term
  crf.setUnaryEnergy( logistic_transform, logistic_feature );
	
  // Add simple pairwise potts terms
  crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 1 ) );
  // Add a longer range label compatibility term
  crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
  //crf.addPairwiseBilateral( 1, 1, 1, 1, 1, im, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	
  // Choose your loss function
  // 	LogLikelihood objective( labeling, 0.01 ); // Log likelihood loss
  // 	Hamming objective( labeling, 0.0 ); // Global accuracy
  // 	Hamming objective( labeling, 1.0 ); // Class average accuracy
  // 	Hamming objective( labeling, 0.2 ); // Hamming loss close to intersection over union
  IntersectionOverUnion objective( labeling ); // Intersection over union accuracy
	
  int NIT = 5;
  const bool verbose = true;
	
  MatrixXf learning_params( 3, 3 );
  // Optimize the CRF in 3 phases:
  //  * First unary only
  //  * Unary and pairwise
  //  * Full CRF
  learning_params<<1,0,0,
    1,1,0,
    1,1,1;

  std::cout<<"Unary parameters: "<<crf.unaryParameters().transpose()<<std::endl;
  std::cout<<"Pairwise parameters: "<<crf.labelCompatibilityParameters().transpose()<<std::endl;
  std::cout<<"Kernel parameters: "<<crf.kernelParameters().transpose()<<std::endl;
	
  for( int i=0; i<learning_params.rows(); i++ ) {
    // Setup the energy
    CRFEnergy energy( crf, objective, NIT, learning_params(i,0), learning_params(i,1), learning_params(i,2) );
    energy.setL2Norm( 1e-3 );
		
    // Minimize the energy
    VectorXf p = minimizeLBFGS( energy, 2, true );
		
    // Save the values
    int id = 0;
    if( learning_params(i,0) ) {
      crf.setUnaryParameters( p.segment( id, crf.unaryParameters().rows() ) );
      id += crf.unaryParameters().rows();
    }
    if( learning_params(i,1) ) {
      crf.setLabelCompatibilityParameters( p.segment( id, crf.labelCompatibilityParameters().rows() ) );
      id += crf.labelCompatibilityParameters().rows();
    }
    if( learning_params(i,2) )
      crf.setKernelParameters( p.segment( id, crf.kernelParameters().rows() ) );
  }
  // Return the parameters
  std::cout<<"Unary parameters: "<<crf.unaryParameters().transpose()<<std::endl;
  std::cout<<"Pairwise parameters: "<<crf.labelCompatibilityParameters().transpose()<<std::endl;
  std::cout<<"Kernel parameters: "<<crf.kernelParameters().transpose()<<std::endl;
	
  // Do map inference
  VectorXs map = crf.map(NIT);
	
  // Store the result
  unsigned char *res = colorize( map, W, H );
  writePPM( argv[3], W, H, res );
	
  delete[] im;
  delete[] anno;
  delete[] res;
}
