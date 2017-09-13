module Neural
  def self.default_activation
    Neural::ActivationFunctions::Logistic.instance
  end
end
