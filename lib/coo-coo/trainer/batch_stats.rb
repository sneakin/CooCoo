module CooCoo
  module Trainer
    class BatchStats
      attr_reader :trainer
      attr_reader :batch
      attr_reader :batch_size
      attr_reader :total_time
      attr_reader :total_loss
      
      def initialize(trainer, batch, batch_size, total_time, total_loss)
        @trainer = trainer
        @batch = batch
        @batch_size = batch_size
        @total_time = total_time
        @total_loss = total_loss
      end

      def average_time
        total_time / batch_size.to_f
      end

      def average_loss
        total_loss / batch_size.to_f
      end
    end
  end
end
