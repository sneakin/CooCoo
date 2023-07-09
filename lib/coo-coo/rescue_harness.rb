module CooCoo
  module RescueHarness
    class << self
      attr_accessor :exception, :action, :value
    end
    
    def self.catch ex, binding
      $ex = self.exception = ex
      self.action = nil
      binding.pry
    end
  end
  
  def self.rescue_harness &block
    block.call
  rescue
    raise unless ENV['COOCOO_RESCUE'] != '0'
    RescueHarness.catch($!, block.binding)
    case RescueHarness.action
    when :retry then retry
    when :return then RescueHarness.value
    when :exit then exit(RescueHarness.value || -1)
    else raise
    end
  end
end
