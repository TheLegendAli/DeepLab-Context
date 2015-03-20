// Liang-Chieh Chen
//

#include <cstdio>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>
#include "matio.h"
#include "util.h"

template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

void LoadMatFile(const std::string& fn, MatrixXf& data, const int row, const int col,
		 int* channel) {
  mat_t *matfp;
  matfp = Mat_Open(fn.c_str(), MAT_ACC_RDONLY);
  if (matfp == NULL) {
    std::cerr << "Error opening MAT file " << fn;
  }

  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp,"data");
  if (matvar == NULL) {
    std::cerr << "Field 'data' not present in MAT file " << fn << std::endl;
  }

  if (matvar->class_type != matio_class_map<float>()) {
    std::cerr << "Field 'data' must be of the right class (single) in MAT file " << fn << std::endl;
  }
  if (matvar->rank >= 4) {
    if (matvar->dims[3] != 1) {
      std::cerr << "Rank: " << matvar->rank << ". Field 'data' cannot have ndims > 3 in MAT file " << fn << std::endl;
    }
  }

  int file_size = 1;
  int data_size = row * col;
  for (int k = 0; k < matvar->rank; ++k) {
    file_size *= matvar->dims[k];
    
    if (k > 1) {
      data_size *= matvar->dims[k];
    }
  }
  
  assert(data_size <= file_size);

  int data_channel = static_cast<int>(matvar->dims[2]);
  float* file_data = new float[file_size];
  data.resize(data_channel, row*col);
  
  int ret = Mat_VarReadDataLinear(matfp, matvar, file_data, 0, 1, file_size);
  if (ret != 0) {
    std::cerr << "Error reading array 'data' from MAT file " << fn << std::endl;
    return;
  }

  // matvar->dims[0] : width
  // matvar->dims[1] : height
  int in_offset = matvar->dims[0] * matvar->dims[1];
  int in_ind, out_ind;

  // extract from file_data
  for (int c = 0; c < data_channel; ++c) {
    for (int m = 0; m < row; ++m) {
      for (int n = 0; n < col; ++n) {
	out_ind = n + m * col;  // row-order

	// perform transpose of file_data
	in_ind  = n + m * matvar->dims[1];  // col-order but transpose

	// note the minus sign
	data(c, out_ind) = -file_data[in_ind + c*in_offset];  
      }
    }
  }
  
  if(channel != NULL) {
    *channel = data_channel;
  }  


  Mat_VarFree(matvar);
  Mat_Close(matfp);

  delete[] file_data;
}

template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
    int* row, int* col, int* channel) {
  std::ifstream ifs(fn.c_str(), std::ios_base::in | std::ios_base::binary);

  if(!ifs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }

  int num_row, num_col, num_channel;

  ifs.read((char*)&num_row, sizeof(int));
  ifs.read((char*)&num_col, sizeof(int));
  ifs.read((char*)&num_channel, sizeof(int));

  int num_el;

  num_el = num_row * num_col * num_channel;

  data = new T[num_el];

  ifs.read((char*)&data[0], sizeof(T)*num_el);

  ifs.close();

  if(row!=NULL) {
    *row = num_row;
  }

  if(col!=NULL) {
    *col = num_col;
  }
 
  if(channel != NULL) {
    *channel = num_channel;
  }

}

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
    int row, int col, int channel) {
  std::ofstream ofs(fn.c_str(), std::ios_base::out | std::ios_base::binary);

  if(!ofs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }  

  ofs.write((char*)&row, sizeof(int));
  ofs.write((char*)&col, sizeof(int));
  ofs.write((char*)&channel, sizeof(int));

  int num_el;

  num_el = row * col * channel;

  ofs.write((char*)&data[0], sizeof(T)*num_el);

  ofs.close();
}

void TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames) {
  DIR *dir, *tstdp;
  struct dirent *dp;

  //open the directory
  if((dir  = opendir(path.c_str())) == NULL) {
    std::cout << "Error opening " << path << std::endl;
    return;
  }

  while ((dp = readdir(dir)) != NULL) {
    tstdp=opendir(dp->d_name);
		
    if(tstdp) {
      closedir(tstdp);
      if(subdirectories) {
	//TraverseDirectory(
      }
    } else {
      if(fnmatch(pattern.c_str(), dp->d_name, 0)==0) {
	//std::string tmp(path);	
	//tmp.append("/").append(dp->d_name);
	//fileNames.push_back(tmp);  //assume string ends with .bin

	std::string tmp(dp->d_name);
	fileNames.push_back(tmp.substr(0, tmp.length()-4));

	//std::cout << fileNames.back() << std::endl;
      }
    }
  }

  closedir(dir);
  return;
}

void OutputSetting(const InputData& inp) {
  std::cout << "Input Parameters: " << std::endl;
  std::cout << "ImgDir:           " << (inp.ImgDir?inp.ImgDir:"") << std::endl;
  std::cout << "GtDir:            " << (inp.GtDir?inp.GtDir:"") << std::endl;
  std::cout << "FeatureDir:       " << (inp.FeatureDir?inp.FeatureDir:"") << std::endl;
  std::cout << "SaveDir:          " << (inp.SaveDir?inp.SaveDir:"") << std::endl;
  std::cout << "ModelDir:         " << (inp.ModelDir?inp.ModelDir:"") << std::endl;
  std::cout << "Model:            " << (inp.Model?inp.Model:"") << std::endl;

  std::cout << "MaxIterations:    " << inp.MaxIterations << std::endl;
  std::cout << "PosW:      " << inp.PosW    << std::endl;
  std::cout << "PosXStd:   " << inp.PosXStd << std::endl;
  std::cout << "PosYStd:   " << inp.PosYStd << std::endl;
  std::cout << "Bi_W:      " << inp.BilateralW    << std::endl;
  std::cout << "Bi_X_Std:  " << inp.BilateralXStd << std::endl;
  std::cout << "Bi_Y_Std:  " << inp.BilateralYStd << std::endl;
  std::cout << "Bi_R_Std:  " << inp.BilateralRStd << std::endl;
  std::cout << "Bi_G_Std:  " << inp.BilateralGStd << std::endl;
  std::cout << "Bi_B_Std:  " << inp.BilateralBStd << std::endl;  

  switch (inp.Task) {
  case 0:
    std::cout << "Task: LEARNING  " << std::endl;
    break;
  case 1:
    std::cout << "Task: INFERENCE " << std::endl;
    break;
  default:
    std::cerr << "Wrong Task value..." << std::endl;
    break;
  }

  std::cout << "OptimizePhase:  " << inp.OptimizePhase << std::endl;  


  switch(inp.ModelType) {
  case 0:
    std::cout << "Model type: Potts " << std::endl;
    break;
  case 1:
    std::cout << "Model type: Diagonal " << std::endl;
    break;
  case 2:
    std::cout << "Model type: Matrix " << std::endl;
    break;
  default:
    std::cerr << "Wrong Model Type..." << std::endl;
    break;
  }

  std::cout << "Epoch:          " << inp.Epoch << std::endl;
  std::cout << "BatchSize:      " << inp.BatchSize << std::endl;
  std::cout << "L2Norm:         " << inp.L2Norm << std::endl;
  std::cout << "LBFGSItr        " << inp.LBFGSItr << std::endl;
  std::cout << "RandomShuffle   " << inp.RandomShuffle << std::endl;  
  std::cout << "Verbose:        " << inp.Verbose << std::endl;
}

int ParseInput(int argc, char** argv, struct InputData& OD) {
  for(int k=1;k<argc;++k) {
    if(::strcmp(argv[k], "-id")==0 && k+1!=argc) {
      OD.ImgDir = argv[++k];
    } else if(::strcmp(argv[k], "-gd")==0 && k+1!=argc) {
      OD.GtDir = argv[++k];
    } else if(::strcmp(argv[k], "-fd")==0 && k+1!=argc) {
      OD.FeatureDir = argv[++k];
    } else if(::strcmp(argv[k], "-sd")==0 && k+1!=argc) {
      OD.SaveDir = argv[++k];
    } else if(::strcmp(argv[k], "-md")==0 && k+1!=argc) {
      OD.ModelDir = argv[++k];
    } else if(::strcmp(argv[k], "-m")==0 && k+1!=argc) {
      OD.Model = argv[++k];
    } else if(::strcmp(argv[k], "-i")==0 && k+1!=argc) {
      OD.MaxIterations = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-px")==0 && k+1!=argc) {
      OD.PosXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-py")==0 && k+1!=argc) {
      OD.PosYStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-pw")==0 && k+1!=argc) {
      OD.PosW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bx")==0 && k+1!=argc) {
      OD.BilateralXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-by")==0 && k+1!=argc) {
      OD.BilateralYStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bw")==0 && k+1!=argc) {
      OD.BilateralW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-br")==0 && k+1!=argc) {
      OD.BilateralRStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bg")==0 && k+1!=argc) {
      OD.BilateralGStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bb")==0 && k+1!=argc) {
      OD.BilateralBStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-t")==0 && k+1!=argc) {
      OD.Task = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-o")==0 && k+1!=argc) {
      OD.OptimizePhase = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-e")==0 && k+1!=argc) {
      OD.Epoch = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-bs")==0 && k+1!=argc) {
      OD.BatchSize = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-li")==0 && k+1!=argc) {
      OD.LBFGSItr = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-l2")==0 && k+1!=argc) {
      OD.L2Norm = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-v")==0 && k+1!=argc) {
      OD.Verbose = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-rs")==0 && k+1!=argc) {
      OD.RandomShuffle = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-mt")==0 && k+1!=argc) {
      OD.ModelType = atoi(argv[++k]);
    } 
  }
  return 0;
}

void ReshapeToMatlabFormat(short* result, VectorXs& map, int img_row, int img_col) {
  //row-order to column-order

  int out_index, in_index;

  for (int h = 0; h < img_row; ++h) {
    for (int w = 0; w < img_col; ++w) {
      out_index = w * img_row + h;
      in_index  = h * img_col + w;
      result[out_index] = map[in_index];
    }
  }
}

void ComputeUnaryForCRF(float*& unary, float* feat, int feat_row, int feat_col, int feat_channel) {
  int out_index, in_index;

  for (int h = 0; h < feat_row; ++h) {
    for (int w = 0; w < feat_col; ++w) {
      for (int c = 0; c < feat_channel; ++c) {
	out_index = (h * feat_col + w) * feat_channel + c;
	in_index  = (c * feat_col + w) * feat_row + h;
	//unary[out_index] = -log(feat[in_index]);
	unary[out_index] = -feat[in_index];
      }
    }
  }
}

void GetImgNamesFromFeatFiles(std::vector<std::string>& out, const std::vector<std::string>& in, const std::string& strip_pattern) {
  out.clear();
  for (size_t k = 0; k < in.size(); ++k) {
    size_t pos = in[k].find(strip_pattern);
    if (pos != std::string::npos) {
      out.push_back(in[k].substr(0, pos));      
    }
  }
}

void GetLabeling(VectorXs& labeling, const unsigned char* im, const int num_ele) {
  labeling.resize(num_ele);

  for (int k = 0; k < num_ele; ++k) {
    labeling[k] = (int)im[k];
  }
}

template 
void SaveBinFile(std::string& fn, short* data, 
		 int row, int col, int channel);



/*
// read/write Eigen Matrix from/to binary files
template<class Matrix>
void WriteEigenBinary(const char* filename, const Matrix& matrix) {
    std::ofstream out(filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void LoadEigenBinary(const char* filename, Matrix& matrix){
  std::ifstream in(filename, std::ios_base::in | std::ios_base::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}
*/
