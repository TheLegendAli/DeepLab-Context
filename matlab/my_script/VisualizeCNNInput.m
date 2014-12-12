fn = '../../img.bin';

data = LoadBinFile(fn, 'single');
data = data / 255;

fn = '../../seg.bin';
seg = LoadBinFile(fn, 'single');

