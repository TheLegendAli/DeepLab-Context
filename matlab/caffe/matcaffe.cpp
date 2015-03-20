//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static shared_ptr<SGDSolver<float> > solver_;
static vector<shared_ptr<Net<float> > > net_;
static vector<int> init_key;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom, int model_idx) {
  vector<Blob<float>*>& input_blobs = net_[model_idx]->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetNumberOfElements(bottom)),
    input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    CHECK(mxIsSingle(elem))
      << "MatCaffe require single-precision float point data";
    CHECK_EQ(mxGetNumberOfElements(elem), input_blobs[i]->count())
      << "MatCaffe input size does not match the input size of the network";
    const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
        input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
        input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_[model_idx]->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->cpu_data(),
        data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(),
        data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_backward(const mxArray* const top_diff, int model_idx) {
  vector<Blob<float>*>& output_blobs = net_[model_idx]->output_blobs();
  vector<Blob<float>*>& input_blobs = net_[model_idx]->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
    output_blobs.size());
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
        output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
        output_blobs[i]->mutable_gpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_[model_idx]->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights(shared_ptr<Net<float> > net) {
  const vector<shared_ptr<Layer<float> > >& layers = net->layers();
  const vector<string>& layer_names = net->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
          mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
          layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
            weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
            weights_ptr);
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}

void set_input_size(MEX_ARGS){
  if (nrhs != 5)
  {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int height = mxGetScalar(prhs[0]);
  int width = mxGetScalar(prhs[1]);
  int channel = mxGetScalar(prhs[2]);
  int num = mxGetScalar(prhs[3]);
  int layer = mxGetScalar(prhs[4]);
  int model_idx = 0;

  if (net_[model_idx]->layers().size() <= 0)
    mexErrMsgTxt("caffe_mex : set_input_size :: no layer loaded.\n");

  if (net_[model_idx]->input_blobs().size() <= 0)
    mexErrMsgTxt("caffe_mex : set_input_size :: first layer has no input.\n");

  if (layer < 0) {
    for (int i = 0; i < net_[model_idx]->input_blobs().size(); ++ i)
      net_[model_idx]->input_blobs()[i]->Reshape(num, channel, height, width);
  } else {
    if (net_[model_idx]->input_blobs().size() <= layer) {
      mexErrMsgTxt("caffe_mex : set_input_size :: unknown layer.\n");
    } else {
      net_[model_idx]->input_blobs()[layer]->Reshape(num, channel, height, width);
    }
  }
  for (int i = 0; i < net_[model_idx]->layers().size(); ++i)
    net_[model_idx]->layers()[i]->Reshape(net_[model_idx]->bottom_vecs()[i], net_[model_idx]->top_vecs()[i]);
}

static void get_weights(MEX_ARGS) {
  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_get_weights(net_[model_idx]);
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void set_random_seed(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int random_seed = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::set_random_seed(random_seed);
}


static void get_init_key(MEX_ARGS) {

  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = mxCreateDoubleScalar(init_key[model_idx]);
}

static void init(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);

  int model_idx = 0;
  if (nrhs > 2)
    model_idx = (int)mxGetScalar(prhs[2]);

  if (net_.size() <= model_idx)
  {
    net_.resize(model_idx + 1);
    init_key.resize(model_idx + 1);
  }

  net_[model_idx].reset(new Net<float>(string(param_file)));
  mxFree(param_file);

  if (nrhs > 1) {
    char* model_file = mxArrayToString(prhs[1]);
    net_[model_idx]->CopyTrainedLayersFrom(string(model_file));
    mxFree(model_file);
  }

  init_key[model_idx] = (int)caffe_rng_rand();  // NOLINT(caffe/random_fn)
  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key[model_idx]);
  }
}

static void save_weights(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* model_file = mxArrayToString(prhs[0]);
  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (net_.size() <= model_idx)
  {
    mexErrMsgTxt("caffe_mex : Wrong model_idx");
  }

  NetParameter net_param;
  net_[model_idx]->ToProto(&net_param, false);

  LOG(INFO) << "Snapshotting to " << model_file;
  WriteProtoToBinaryFile(net_param, model_file);
}

static void release(MEX_ARGS) {
  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  if (net_[model_idx]) {
    net_[model_idx].reset();
    init_key[model_idx] = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_forward(prhs[0], model_idx);
}


static void backward(MEX_ARGS) {
  if (nrhs != 1 && nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  int model_idx = 0;
  if (nrhs > 1)
    model_idx = (int)mxGetScalar(prhs[1]);

  if (model_idx >= net_.size())
    mexErrMsgTxt("caffe_mex : Un-inited net");

  plhs[0] = do_backward(prhs[0], model_idx);
}

static void is_initialized(MEX_ARGS) {
  int model_idx = 0;
  if (nrhs > 0)
    model_idx = (int)mxGetScalar(prhs[0]);


  if (net_.size() < model_idx || !net_[model_idx]) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void read_mean(MEX_ARGS) {
  if (nrhs != 1) {
    mexErrMsgTxt("caffe_mex : Usage: caffe('read_mean', 'path_to_binary_mean_file'");
    return;
  }
  const string& mean_file = mxArrayToString(prhs[0]);
  Blob<float> data_mean;
  LOG(INFO) << "Loading mean file from" << mean_file;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
  if (!result) {
    mexErrMsgTxt("caffe_mex : Couldn't read the file");
    return;
  }
  data_mean.FromProto(blob_proto);
  mwSize dims[4] = {data_mean.width(), data_mean.height(),
    data_mean.channels(), data_mean.num() };
  mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
  caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
  mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
    " format and channels are also BGR!");
  plhs[0] = mx_blob;
}

/*
static void init_solver(MEX_ARGS) {
  if (nrhs < 1 || nrhs >= 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (solver_.use_count() != 0){
    mexPrintf("caffe_mex : Exist another solver, init failed!\n");
    return;
  }

  char* solver_file = mxArrayToString(prhs[0]);

  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  LOG(INFO) << "Starting Optimization";
  solver_.reset(new SGDSolver<float>(solver_param));

  if (nrhs >= 2) {
    char* model_file = mxArrayToString(prhs[1]);

    if (!string(model_file).empty() ){
      LOG(INFO) << "Recovery from " << model_file;
      solver_->net()->CopyTrainedLayersFrom(string(model_file));  
    }
    
    mxFree(model_file);
  }
  solver_->PreSolve();

  mxFree(solver_file);
}

static void recovery_solver(MEX_ARGS) {
  if (nrhs < 1 || nrhs > 3) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  char* model_file = mxArrayToString(prhs[1]);
  if ( string(model_file).empty() ){
    LOG(ERROR) << "No recovery model";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }


  char* solver_file = mxArrayToString(prhs[0]);

  if (nrhs >= 3){
    ::google::InitGoogleLogging("caffe_mex");
    char* log_file = mxArrayToString(prhs[2]);
#ifdef WIN32
    int mkd_result = _mkdir(log_file);
    CHECK_NE(mkd_result, 0) << "Failed to open a temporary directory at: " << log_file;
#else
    char* mkd_result = mkdtemp(log_file);
    CHECK(mkd_result != NULL)
      << "Failed to create a temporary directory at: " << mkd_result;
#endif
    ::google::SetLogDestination(0, log_file);
    mxFree(log_file);
  }


  LOG(INFO) << "Loading from " << solver_file;
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);

  LOG(INFO) << "Starting Optimization";
  solver_.reset(new SGDSolver<float>(solver_param));
  solver_->PreSolve();

  LOG(INFO) << "Restoring previous solver status from " << model_file;
  printf("Resuming form %s\n", model_file);
  solver_->Restore(model_file);

  mxFree(model_file);
  mxFree(solver_file);
}

static void release_solver(MEX_ARGS) {
  solver_.reset();
}

static mxArray* do_train(const mxArray* const bottom) {

  if (solver_.use_count() == 0)
  {
    mexPrintf("caffe_mex : No solver inited!\n");
    return mxCreateDoubleMatrix(0, 0, mxREAL);
  }

  //Caffe::set_phase(Caffe::TRAIN);
  if (Caffe::phase() != Caffe::TRAIN)
    mexErrMsgTxt("caffe_mex : caffe:train must in train phase, please set phase to be train.\n");

  vector<Blob<float>*>& input_blobs = solver_->net()->input_blobs();
  CHECK_EQ(mxGetNumberOfElements(bottom), input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    CHECK(mxIsSingle(elem));
    CHECK_EQ(mxGetNumberOfElements(elem), input_blobs[i]->count());
    const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  float error, loss;
  solver_->SolveOneIterationPrefilled(error, loss);

  mxArray* mx_out = mxCreateDoubleMatrix(2, 1, mxREAL);
  double* data_ptr = reinterpret_cast<double*>(mxGetPr(mx_out));
  data_ptr[0] = error;
  data_ptr[1] = loss;

  return mx_out;
}

static void train(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  plhs[0] = do_train(prhs[0]);
}

*/

static mxArray* do_test(const mxArray* const bottom) {

  if (solver_.use_count() == 0)
  {
    mexPrintf("caffe_mex : No solver inited!\n");
    return mxCreateDoubleMatrix(0, 0, mxREAL);
  }

  Caffe::set_phase(Caffe::TEST);
  vector<Blob<float>*>& input_blobs = solver_->net()->input_blobs();
  //CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
  //input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    const float* const data_ptr =
      reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  float error;
  const vector<Blob<float>*>& result = solver_->net()->ForwardPrefilled();

  vector<float> test_score;
  for (int j = 0; j < result.size(); ++j) {
    const float* result_vec = result[j]->cpu_data();
    for (int k = 0; k < result[j]->count(); ++k) {
      test_score.push_back(result_vec[k]);
    }
  }
  error = 1 - test_score[0];

  mxArray* mx_out = mxCreateDoubleMatrix(1, 1, mxREAL);
  double* data_ptr = reinterpret_cast<double*>(mxGetPr(mx_out));
  data_ptr[0] = error;

  Caffe::set_phase(Caffe::TRAIN);
  return mx_out;
}

static void test(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  plhs[0] = do_test(prhs[0]);
}

/*
static void get_solver_iter(MEX_ARGS) {

  if (solver_.use_count() == 0)
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  plhs[0] = mxCreateDoubleScalar(solver_->Iter());
}

static void snapshot(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (solver_.use_count() == 0)
  {
    mexPrintf("No solver inited!\n");
    return;
  }

  char* filename = mxArrayToString(prhs[0]);

  solver_->Snapshot("", filename);
}
*/

static void do_set_weights(shared_ptr<Net<float> > net, const mxArray* mx_layers) {
  const vector<shared_ptr<Layer<float> > >& layers = net->layers();
  const vector<string>& layer_names = net->layer_names();

  unsigned int input_layer_num = mxGetNumberOfElements(mx_layers);
  for (unsigned int i = 0; i < input_layer_num; ++i)
  {

    // Step 1: get input layer information
    mxArray *mx_layer_name = mxGetField(mx_layers, i, "layer_names");
    if (mx_layer_name == NULL)
    {
      mexPrintf("layer %d has no field ""layer_names"", ignore\n", i);
      continue;
    }
    char *layer_name = mxArrayToString(mx_layer_name);
    mxArray *mx_weights_cell = mxGetField(mx_layers, i, "weights");
    if (mx_weights_cell == NULL)
    {
      mexPrintf("layer %d has no field ""weights"", ignore\n", i);
      continue;
    }
    if (!mxIsCell(mx_weights_cell))
    {
      mexPrintf("layer %d field ""weights"" is not cell, ignore\n", i);
      continue;
    }
    unsigned int weight_blob_num = mxGetNumberOfElements(mx_weights_cell);


    // Step 2: scan model layers, and try to set layer
    string prev_layer_name = "";
    for (unsigned int j = 0; j < layers.size(); ++j) {

      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[j]->blobs();
      if (layer_blobs.size() == 0)
        continue;

      if (layer_names[j] != string(layer_name))
        continue;

      if (weight_blob_num != layer_blobs.size())
      {
        mexPrintf("%s has %d blobs, while model layer has %d blobs, ignore\n", layer_name, weight_blob_num, layer_blobs.size());
        continue;
      }

      for (unsigned int k = 0; k < layer_blobs.size(); ++k) {
        bool setted = false;
        mxArray *mx_weights = mxGetCell(mx_weights_cell, k);
        const mwSize* input_blob_dims = mxGetDimensions(mx_weights);
        int dim_num = mxGetNumberOfDimensions(mx_weights);
        size_t input_dims[4] = {1, 1, 1, 1};
        for (int idim = 0; idim < dim_num; ++idim)
          input_dims[idim] = input_blob_dims[idim];

        if (layer_blobs[k]->width() != (int)input_dims[0] || layer_blobs[k]->height() != (int)input_dims[1] || layer_blobs[k]->channels() != (int)input_dims[2] || layer_blobs[k]->num() != (int)input_dims[3])
        {
          mexPrintf("%s blobs %d dims don't match, ignore\n", layer_name, k);
          continue;
        }

        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));
        
        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[k]->count(), weights_ptr, layer_blobs[k]->mutable_cpu_data());
          setted = true;
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[k]->count(), weights_ptr, layer_blobs[k]->mutable_gpu_data());
          setted = true;
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }

        if (setted)
          mexPrintf("copied weights for %s blob %d \n", layer_name, k);
      }

    }
    mxFree(layer_name);
  }
}


static void set_weights(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  do_set_weights(net_[0], prhs[0]);
}


static void get_weights_solver(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (solver_.use_count() == 0)
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  plhs[0] = do_get_weights(solver_->net());
}

static void set_weights_solver(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("caffe_mex : Wrong number of arguments");
  }

  if (solver_.use_count() == 0)
  {
    mexPrintf("No solver inited!\n");
    plhs[0] = mxCreateDoubleScalar(-1);
    return;
  }

  do_set_weights(solver_->net(), prhs[0]);
}


/** -----------------------------------------------------------------
** Available commands.
**/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward       },
  { "backward",           backward      },
  { "init",               init        },
  { "is_initialized",     is_initialized    },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train   },
  { "set_phase_test",     set_phase_test    },
  { "set_device",         set_device      },
  { "set_input_size",     set_input_size    },
  { "get_weights",        get_weights     },
  { "set_weights",        set_weights     },
  { "save_weights",       save_weights     },
  { "get_init_key",       get_init_key    },
  { "release",            release       },
  { "read_mean",          read_mean     },
  { "set_random_seed",  set_random_seed   },
  // for solver
  //{ "init_solver",    init_solver     },
  //{ "recovery_solver",  recovery_solver   },
  //{ "release_solver",   release_solver    },
  //{ "get_solver_iter",  get_solver_iter   },
  { "get_weights_solver", get_weights_solver  },
  { "set_weights_solver", set_weights_solver  }, 
  //{ "train",        train       },
  { "test",       test        },
  //{ "snapshot",     snapshot      },
  // The end.
  { "END",                NULL        },
};


/** -----------------------------------------------------------------
** matlab entry point: caffe(api_command, arg1, arg2, ...)
**/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "caffe_mex : No API command given";
    mexErrMsgTxt("caffe_mex : An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("caffe_mex : API command not recognized");
    }
    mxFree(cmd);
  }
}
