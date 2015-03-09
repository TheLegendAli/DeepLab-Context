/*
 * The code is modified from the ICML demo code by Philipp Krähenbühl
 *
 */

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
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>

//#include <string.h>
//#include <fstream>
//#include <dirent.h>
//#include <fnmatch.h>

#include "ppm.h"
//#include "common.h"

#include "lodepng.h"
//#include "matio.h"
#include "util.h"

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
  InputData inp;

  ParseInput(argc, argv, inp);
  OutputSetting(inp);

  assert(inp.ImgDir != NULL && inp.FeatureDir != NULL && inp.SaveDir != NULL);
  assert(inp.Task >=0 && inp.Task <= 1);
  assert(inp.OptimizePhase >= 1 && inp.OptimizePhase <= 3);
  if (inp.Task == 1) {
    assert(inp.Epoch == 1);
    if (inp.ModelDir == NULL || inp.Model == NULL) {
     std::cout << "Using default values for inference ..." << std::endl;
    }
  }

  std::string pattern = "*.mat";
  std::vector<std::string> feat_file_names;
  std::string feat_folder(inp.FeatureDir);
  TraverseDirectory(feat_folder, pattern, false, feat_file_names);
  
  std::string strip_pattern("_blob_0");
  std::vector<std::string> img_file_names;
  GetImgNamesFromFeatFiles(img_file_names, feat_file_names, strip_pattern);

  std::string fn;
  unsigned char* img = NULL;
  unsigned char* gt  = NULL;

  unsigned int gt_row, gt_col;
  VectorXs labeling;

  int feat_row, feat_col, feat_channel;
  unsigned error_code;
  bool init_param = true;

  VectorXf crf_params;
  MatrixXi learning_params(inp.OptimizePhase, 3);
  // Optimize the CRF in 3 phases:
  //  * First unary only
  //  * Unary and pairwise
  //  * Full CRF
  if (inp.OptimizePhase == 1) {
    learning_params << 1, 0, 0;
  }
  if (inp.OptimizePhase == 2) {
    learning_params << 1, 0, 0,
      1, 1, 0;
  }
  if (inp.OptimizePhase == 3) {
    learning_params << 1, 0, 0,
      1, 1, 0,
      1, 1, 1;
  }


  for (int e = 0; e < inp.Epoch; ++e) {
    for (size_t k = 0; k < img_file_names.size(); ++k) {
      if ((k+1) % 100 == 0) {
	std::cout << "Epoch " << e << ": processing " << k+1 << " (" << img_file_names.size() << ")" << std::endl;
      }

      // read ppm image
      fn = std::string(inp.ImgDir) + "/" + img_file_names[k] + ".ppm";
      img = readPPM(fn.c_str(), feat_col, feat_row);
      if (!img){
	std::cerr << "Failed to load image!" << std::endl;
	return 1;
      }


      // read mat features
      MatrixXf feat;
      fn = std::string(inp.FeatureDir) + "/" + feat_file_names[k] + ".mat";
      LoadMatFile(fn, feat, feat_row, feat_col, &feat_channel);

      // Setup the CRF model
      DenseCRF2D crf(feat_col, feat_row, feat_channel);

      crf.setUnaryEnergy(feat);

      // Add simple pairwise potts terms
      crf.addPairwiseGaussian(inp.PosXStd, inp.PosYStd, 
			      new PottsCompatibility(inp.PosW));
      // Add a longer range label compatibility term
      /*
	crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	       inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
			       new PottsCompatibility(inp.BilateralW));
      */
      ///*
      // note the minus sign for MatrixCompatibility
      // if use Potts, just use plus sign
      ///*
      crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	       inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
	       new MatrixCompatibility(-inp.BilateralW * 
		       MatrixXf::Identity(feat_channel, feat_channel))); 
      //*/
      /*
      crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	       inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
	       new DiagonalCompatibility(-inp.BilateralW * 
		       VectorXf::Ones(feat_channel, 1))); 
      */


      //*/
      if (inp.Task == 0) {
	if (e != 0 || k != 0) {
	  // load trained parameters
	  crf.SetParameters(crf_params, 
			    learning_params.row(learning_params.rows()-1));
	} //else {
	  std::cout << "Init Unary parameters: " << 
	    crf.unaryParameters().transpose() << std::endl;
	  std::cout << "Init Pairwise parameters: " << 
	    crf.labelCompatibilityParameters().transpose() << std::endl;
	  std::cout << "Init Kernel parameters: " << 
	    crf.kernelParameters().transpose() << std::endl;
	  //}

	// read png ground truth
  	fn = std::string(inp.GtDir) + "/" + img_file_names[k] + ".png";
	error_code = lodepng_decode_file(&gt, &gt_col, &gt_row, fn.c_str(), LCT_GREY, 8);
	if (error_code) {
	  std::cerr << "decoder error " << error_code << ": " << lodepng_error_text(error_code) << std::endl;
	  return 1;
	}

	assert(feat_col == (int)gt_col && feat_row == (int)gt_row);
	GetLabeling(labeling, gt, gt_row*gt_col);
  
	// Choose your loss function
	//LogLikelihood objective( labeling, 0.01 ); // Log likelihood loss
	// 	Hamming objective( labeling, 0.0 ); // Global accuracy
	// 	Hamming objective( labeling, 1.0 ); // Class average accuracy
	//      Hamming objective( labeling, 0.2 ); // Hamming loss close to intersection over union
	
	// Intersection over union accuracy
	IntersectionOverUnion objective( labeling );
	
        //std::cout << "5" << std::endl;
	for( int i=0; i<learning_params.rows(); i++ ) {
	  // Setup the energy
	  CRFEnergy energy( crf, objective, inp.MaxIterations, learning_params(i,0), learning_params(i,1), learning_params(i,2) );
	  energy.setL2Norm( 1e-5 );

	  // Minimize the energy
	  int num_restart = 0;
	  bool lbfgs_verbose = false;
	  crf_params = minimizeLBFGS( energy, num_restart, lbfgs_verbose);

	  // Save the values
	  crf.SetParameters(crf_params, learning_params.row(i));

	} // done learning

	// save parameters
	if (k == img_file_names.size() - 1) {
	  std::cout << "Unary parameters: " <<
	    crf.unaryParameters().transpose() << std::endl;
	  std::cout << "Pairwise parameters: " <<
	    crf.labelCompatibilityParameters().transpose() << std::endl;
	  std::cout << "Kernel parameters: " <<
	    crf.kernelParameters().transpose() << std::endl;

	  std::stringstream ss;
	  ss << (e + 1);
	  std::string epoch;
	  ss >> epoch;

	  fn = std::string(inp.ModelDir) + "/" + std::string(inp.Model) + "_Epoch" + epoch + ".bin";
	  crf.SaveParameters(fn.c_str());
	}
      }
	
      if (inp.Task == 1) { 	
	if (inp.ModelDir != NULL && inp.Model != NULL) {
	  // load parameters
	  fn = std::string(inp.ModelDir) + "/" + std::string(inp.Model) + ".bin";
	  crf.LoadParameters(fn.c_str());
	}

	// Do map inference
	VectorXs map = crf.map(inp.MaxIterations);

	short* result = new short[feat_row*feat_col];
	ReshapeToMatlabFormat(result, map, feat_row, feat_col);

	// save results
	fn = std::string(inp.SaveDir) + "/" + img_file_names[k] + ".bin";
	SaveBinFile(fn, result, feat_row, feat_col, 1);

	delete[] result;  
      }	
	
      delete[] img;
      free(gt);
    }

  }

  

}
