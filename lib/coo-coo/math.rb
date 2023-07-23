require 'coo-coo/core_ext'
require 'coo-coo/math/abstract_vector'
require 'coo-coo/math/functions'
require 'coo-coo/math/interpolation'
require 'coo-coo/math/ruby'

if ENV["COOCOO_USE_CUDA"] != "0"
  begin
    require 'coo-coo/cuda'
    require 'coo-coo/cuda/vector'
  rescue LoadError
  end
end

module CooCoo
  if ENV["COOCOO_USE_CUDA"] != "0" && CooCoo::CUDA.available?
    Vector = CUDA::Vector
  elsif ENV["COOCOO_USE_NMATRIX"] == '1'
    require 'coo-coo/math/nmatrix'
    Vector = NMatrix::Vector
  else
    Vector = Ruby::Vector
  end
end
