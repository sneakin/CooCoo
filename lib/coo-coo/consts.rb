module CooCoo
  def self.default_activation
    CooCoo::ActivationFunctions::Logistic.instance
  end
end
