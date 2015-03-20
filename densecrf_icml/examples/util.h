// Liang-Chieh Chen
//

#ifndef _DENSECRF_UTIL_H_
#define _DENSECRF_UTIL_H_

#include <string>
#include <vector>
#include "densecrf.h"
#include "matio.h"

struct InputData {
  char* ImgDir;
  char* GtDir;
  char* FeatureDir;
  char* SaveDir;
  char* ModelDir;
  char* Model;

  // used to specify the number of message passing for training
  int MaxIterations;

  float PosXStd;
  float PosYStd;
  float PosW;
  float BilateralXStd;
  float BilateralYStd;
  float BilateralRStd;
  float BilateralGStd;
  float BilateralBStd;
  float BilateralW;

  // 0: train, 1: inference, 2: train+inference
  int Task;     

  // 1: optimize unary only
  // 2: optimize both unary and pairwise
  // 3: optimize Full CRF (i.e., also kernel)
  int OptimizePhase;
  int ModelType;  //0: Potts, 1: Diagonal, 2: Matrix
  int Epoch;
  int BatchSize;
  float L2Norm;
  int LBFGSItr;
  int RandomShuffle;
  int Verbose;
  
  InputData() :
    ImgDir(NULL), GtDir(NULL), FeatureDir(NULL), SaveDir(NULL), 
    ModelDir(NULL), Model(NULL), MaxIterations(10), 
    BilateralW(5), BilateralXStd(70), BilateralYStd(70), BilateralRStd(5),
    BilateralGStd(5), BilateralBStd(5), PosW(3), PosXStd(3), PosYStd(3),
    Task(2), OptimizePhase(3), ModelType(0), Epoch(1), BatchSize(1), L2Norm(0.00001),
    LBFGSItr(1), RandomShuffle(0), Verbose(0) {}
};

template <typename Dtype> 
enum matio_classes matio_class_map();

void LoadMatFile(const std::string& fn, MatrixXf& data, const int row, 
		 const int col, int* channel = NULL);
template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
      int* row = NULL, int* col = NULL, int* channel = NULL);

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
      int row = 1, int col = 1, int channel = 1);

void TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames);

void OutputSetting(const InputData& inp);

int ParseInput(int argc, char** argv, struct InputData& OD);

void ReshapeToMatlabFormat(short* result, VectorXs& map, int img_row, int img_col);

void ComputeUnaryForCRF(float*& unary, float* feat, int feat_row, int feat_col, int feat_channel);

void GetImgNamesFromFeatFiles(std::vector<std::string>& out, const std::vector<std::string>& in, const std::string& strip_pattern);

void GetLabeling(VectorXs& labeling, const unsigned char* im, const int num_ele);



#endif
