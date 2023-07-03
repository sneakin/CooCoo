require File.join(File.dirname(__FILE__), '..', 'spec_helper')
require 'coo-coo/core_ext'

describe String do
  describe '#fill_template' do
    it "replaces each $name with the value from a hash" do
      expect('Hello $name. Good ${tod}. You have ${balance}.'.fill_template(name: 'Alice', 'tod' => 'day')).
        to eq('Hello Alice. Good day. You have ${balance}.')
    end
  end
end