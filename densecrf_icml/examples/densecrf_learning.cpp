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

#include <algorithm>    // std::random_shuffle
//#include <ctime>        // std::time

//#include <string.h>
//#include <fstream>
//#include <dirent.h>
//#include <fnmatch.h>

#include "ppm.h"
//#include "common.h"

#include "lodepng.h"
//#include "matio.h"
#include "util.h"
#include "Timer.h"

// The energy object implements an energy function that is minimized using LBFGS
class CRFEnergy: public EnergyFunction {
protected:
  VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
  //DenseCRF & crf_;
  std::vector<DenseCRF*>& crfs_;

  //const ObjectiveFunction & objective_;
  std::vector<ObjectiveFunction*>& objectives_;

  int NIT_;
  bool unary_, pairwise_, kernel_;
  float l2_norm_;
public:
  //CRFEnergy( DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crf_(crf),objective_(objective),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f){
  CRFEnergy( std::vector<DenseCRF*>& crfs, std::vector<ObjectiveFunction*>& objectives, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crfs_(crfs),objectives_(objectives),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f){

    //std::cout << "c 1" << std::endl;

    initial_u_param_ = crfs_[0]->unaryParameters();

    //std::cout << "c 2" << std::endl;

    initial_lbl_param_ = crfs_[0]->labelCompatibilityParameters();

    //std::cout << "c 3" << std::endl;

    initial_knl_param_ = crfs_[0]->kernelParameters();
    /*
    initial_u_param_ = crf_.unaryParameters();
    initial_lbl_param_ = crf_.labelCompatibilityParameters();
    initial_knl_param_ = crf_.kernelParameters();
    */
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
    VectorXf du = 0*initial_u_param_, dl = 0*initial_lbl_param_, dk = 0*initial_knl_param_;
    double r = 0;
    VectorXf tmp;
    tmp.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
    tmp.setZero();

    //std::cout << "dims:" << du.rows() << "," << dl.rows() << "," << dk.rows() << std::endl;

    //std::cout << "before1 dx: " << dx.size() << std::endl << dx << std::endl << std::endl;

    dx = 0 * tmp;

    //std::cout << "gd 1" << std::endl;

    for (size_t k = 0; k < crfs_.size(); ++k) {
      int p = 0;

      //std::cout << "gd 2:" << k << std::endl;

      if (unary_) {
	crfs_[k]->setUnaryParameters( x.segment( p, initial_u_param_.rows() ) );
	p += initial_u_param_.rows();
      }

      //std::cout << "gd 3:" << k << std::endl;

      if (pairwise_) {
	crfs_[k]->setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
	p += initial_lbl_param_.rows();
      }

      //std::cout << "gd 4:" << k << std::endl;

      if (kernel_)
	crfs_[k]->setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );

      //std::cout << "gd 5:" << k << std::endl;

      r += crfs_[k]->gradient( NIT_, *objectives_[k], unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );

      //std::cout << "gd 6:" << k << "." << tmp.size() << std::endl;

      tmp  << -(unary_?du:VectorXf()), -(pairwise_?dl:VectorXf()), -(kernel_?dk:VectorXf());
      //std::cout << "gd 7:" << k << std::endl;

      //std::cout << " t size and d size:" << tmp.size() << "," << dx.size() << std::endl << std::endl;
      //std::cout << "g: " << k << ":" << tmp << std::endl << std::endl;
      //std::cout << "before dx: " << dx << std::endl << std::endl;
      dx += tmp;

      //std::cout << k << ", dx: " << dx.transpose() << std::endl;
    
    }

    //std::cout << "gd 8:"  << std::endl;

    dx = dx.array() / crfs_.size();

    //std::cout << "c size:" << crfs_.size() << std::endl;
    //std::cout << "dx :"  << dx.transpose() << std::endl;

    r = -r;
    if( l2_norm_ > 0 ) {
      dx += l2_norm_ * x;
      r += 0.5*l2_norm_ * (x.dot(x));
    }

    //std::cout << "gd 10:"  << std::endl;
    //std::cout << "final dx: " << dx.transpose() << std::endl;

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
  assert(inp.BatchSize >= 1);
  assert(inp.L2Norm >= 0);

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

  // for random shuffle
  std::srand(0);

  int img_ind;

  CPrecisionTimer CTmr;
  CTmr.Start();
 
  for (int e = 0; e < inp.Epoch; ++e) {

    // random shuffle inputs
    if (inp.RandomShuffle) {
      std::random_shuffle(feat_file_names.begin(), feat_file_names.end());
      GetImgNamesFromFeatFiles(img_file_names, feat_file_names, strip_pattern);

      //for (int aa = 0; aa < feat_file_names.size(); ++aa) {
      //std::cout <<  feat_file_names[aa] << ":" << img_file_names[aa] << std::endl;
      //}
    }

    for (size_t k = 0; k < img_file_names.size() / inp.BatchSize; ++k) {
      std::vector<DenseCRF*> crfs;
      std::vector<ObjectiveFunction*> objectives;

      std::cout << "Epoch " << e << ": processing batch " << k+1 << " (" << img_file_names.size() / inp.BatchSize << ")" << std::endl;

      // first load all the batches to crfs and objectives
      for (int m = 0; m < inp.BatchSize; ++m) {
	//std::cout << "loading " << m << std::endl;

	img_ind = k * inp.BatchSize + m;

	// read ppm image
	fn = std::string(inp.ImgDir) + "/" + img_file_names[img_ind] + ".ppm";
	img = readPPM(fn.c_str(), feat_col, feat_row);
	if (!img){
	  std::cerr << "Failed to load image!" << std::endl;
	  return 1;
	}

	// read mat features
	MatrixXf feat;
	fn = std::string(inp.FeatureDir) + "/" + feat_file_names[img_ind] + ".mat";
	LoadMatFile(fn, feat, feat_row, feat_col, &feat_channel);

	// Setup the CRF model
	DenseCRF2D* crf = new DenseCRF2D(feat_col, feat_row, feat_channel);

	crf->setUnaryEnergy(feat);

	// Add simple pairwise potts terms
	crf->addPairwiseGaussian(inp.PosXStd, inp.PosYStd, 
			      new PottsCompatibility(inp.PosW));
	// Add a longer range label compatibility term
	///*
	if (inp.ModelType == 0) {
	  crf->addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	  inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
	  new PottsCompatibility(inp.BilateralW));
	} else if(inp.ModelType == 1) {
	  crf->addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	    inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
	    new DiagonalCompatibility(-inp.BilateralW *
				    VectorXf::Ones(feat_channel, 1)));
	} else if (inp.ModelType == 2) {
	  // note the minus sign for MatrixCompatibility
	  // if use Potts, just use plus sign
	  crf->addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, 
	       inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd, img, 
	       new MatrixCompatibility(-inp.BilateralW * 
		       MatrixXf::Identity(feat_channel, feat_channel))); 
	}

	delete[] img;
	img = NULL;
	//*/
	if (inp.Task == 0) {
	  if (e != 0 || k != 0) {
	    // load trained parameters
	    crf->SetParameters(crf_params, 
			      learning_params.row(learning_params.rows()-1));
	  } 

	  if (m == 0 && inp.Verbose) {
	    std::cout << "parameters before learning: " << std::endl;
	    crf->PrintParameters();
	  }

	  // read png ground truth
	  fn = std::string(inp.GtDir) + "/" + img_file_names[img_ind] + ".png";
	  error_code = lodepng_decode_file(&gt, &gt_col, &gt_row, fn.c_str(), LCT_GREY, 8);
	  if (error_code) {
	    std::cerr << "decoder error " << error_code << ": " << lodepng_error_text(error_code) << std::endl;
	    return 1;
	  }

	  //std::cout << "02" << std::endl;

	  assert(feat_col == (int)gt_col && feat_row == (int)gt_row);
	  GetLabeling(labeling, gt, gt_row*gt_col);
  
	  //std::cout << "03" << std::endl;

	  free(gt);
	  gt = NULL;

	  //std::cout << "04" << std::endl;

	  // Choose your loss function
	  //LogLikelihood* objective = new LogLikelihood( labeling, 0.01 ); // Log likelihood loss
	  // 	Hamming objective( labeling, 0.0 ); // Global accuracy
	  // 	Hamming objective( labeling, 1.0 ); // Class average accuracy
	  //    Hamming* objective = new Hamming( labeling, 0.2 ); // Hamming loss close to intersection over union
	  // Intersection over union accuracy
	  IntersectionOverUnion* objective = new IntersectionOverUnion( labeling );
	
	  //std::cout << "05" << std::endl;


	  //std::cout << "06" << std::endl;

	  objectives.push_back(objective);	

	  //std::cout << "07" << std::endl;

	}

	crfs.push_back(crf);
      }

      //
      // after the batch is loaded, perform tasks
      //

      //std::cout << "crfs size:" << crfs.size() << std::endl;
      //crfs[0]->PrintParameters();

      if (inp.Task == 0) {
	for( int i=0; i<learning_params.rows(); i++ ) {
	  if (inp.Verbose) {
	    std::cout << "learning params: " << learning_params.row(i) << std::endl;
	  }

	  //std::cout << "1" << std::endl;

	  // Setup the energy
	  CRFEnergy energy( crfs, objectives, inp.MaxIterations, learning_params(i,0), learning_params(i,1), learning_params(i,2) );
	  energy.setL2Norm(inp.L2Norm);

	  //std::cout << "2" << std::endl;

	  // Minimize the energy
	  int num_restart = 0;
	  //bool lbfgs_verbose = false;
	  crf_params = minimizeLBFGS( energy, num_restart, inp.LBFGSItr, inp.Verbose);

	  //std::cout << "3" << std::endl << std::endl;
	  
	} // done learning

	// save parameters
	if (k == (img_file_names.size() / inp.BatchSize) - 1) {	  	  
	  std::cout << "Saving parameters..." << std::endl;
	  // Save the values
	  crfs[0]->SetParameters(crf_params, 
			      learning_params.row(learning_params.rows()-1));

	  if (inp.Verbose) {
	    crfs[0]->PrintParameters();
	  }

	  std::stringstream ss;
	  ss << (e + 1);
	  std::string epoch;
	  ss >> epoch;

	  fn = std::string(inp.ModelDir) + "/" + std::string(inp.Model) + "_Epoch" + epoch + ".bin";
	  crfs[0]->SaveParameters(fn.c_str());
	}	
      }
    
      if (inp.Task == 1) { 
	if (inp.ModelDir != NULL && inp.Model != NULL) {
	  // load parameters
	  fn = std::string(inp.ModelDir) + "/" + std::string(inp.Model) + ".bin";
	  crfs[0]->LoadParameters(fn.c_str());

	  if (e == 0 && k == 0 && inp.Verbose) {
	    std::cout << " Loaded prameters:" << std::endl;
	    crfs[0]->PrintParameters();
	  }
	} else {
	  if (e == 0 && k == 0 && inp.Verbose) {
	    std::cout << "No specified parameters. Use default ones..." << std::endl;
	    crfs[0]->PrintParameters();
	  }	  
	}
	// Do map inference
	VectorXs map = crfs[0]->map(inp.MaxIterations);

	short* result = new short[feat_row*feat_col];
	ReshapeToMatlabFormat(result, map, feat_row, feat_col);

	// save results
	fn = std::string(inp.SaveDir) + "/" + img_file_names[img_ind] + ".bin";
	SaveBinFile(fn, result, feat_row, feat_col, 1);

	delete[] result;  
      }	      

      // delete crf, and objective
      for (size_t mm = 0; mm < crfs.size(); ++mm) {
	delete crfs[mm];
      }
      for (size_t mm = 0; mm < objectives.size(); ++mm) {
	delete objectives[mm];
      }
    }
  }
  
  std::cout << "Time : " << CTmr.Stop() << std::endl;

}
