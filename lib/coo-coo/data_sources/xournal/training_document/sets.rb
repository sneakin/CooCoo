module CooCoo
  module DataSources
    module Xournal
      class TrainingDocument
        # Create a {TrainingDocument} for ASCII characters.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.ascii_trainer(doc = self.new)
          (32...127).each do |c|
            c = c.chr[0]
            doc.add_example(c, [])
          end

          doc
        end

        # Create a {TrainingDocument} for an arbitrary Unicode block.
        # @param starting_offset [Integer] Which Unicode character to start the examples
        # @param number [Integer] The number of characters to place in the document.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.unicode_trainer(starting_offset, number, doc = self.new)
          number.times do |i|
            doc.add_example("" << (starting_offset + i), [])
          end

          doc
        end

        # Create a {TrainingDocument} for Japanese Hiragana, Katakana, and punctuation.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.jp_trainer(doc = self.new)
          unicode_trainer(0x3000, 64, doc)
          unicode_trainer(0x3040, 96, doc)
          unicode_trainer(0x30A0, 96, doc)
          unicode_trainer(0xff00, 16 * 15, doc)
          doc
        end

        # Create a {TrainingDocument} for the CJK block.
        # @param limit [Integer] Optional limit on how many characters to include.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.cjk_trainer(limit = 2000, doc = self.new)
          unicode_trainer(0x4e00, limit)
        end

        # Create a {TrainingDocument} for emoji.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.emoji_trainer(doc = self.new)
          unicode_trainer(0x1F600, 16 * 5, doc)
          unicode_trainer(0x2700, 16 * 12, doc)
        end

        # Create a {TrainingDocument} for math symbols.
        # @param doc [Document] Optional Document used to extract examples.
        # @return [TrainingDocument]
        def self.math_trainer(doc = self.new)
          unicode_trainer(0x2200, 16 * 16, doc)
          unicode_trainer(0x2A00, 16 * 16, doc)
          unicode_trainer(0x2100, 16 * 5, doc)
          unicode_trainer(0x27C0, 16 * 3, doc)
          unicode_trainer(0x2980, 16 * 8, doc)
          unicode_trainer(0x2300, 16 * 16, doc)
          unicode_trainer(0x25A0, 16 * 6, doc)
          unicode_trainer(0x2B00, 16 * 16, doc)
          unicode_trainer(0x2190, 16 * 7, doc)
          unicode_trainer(0x2900, 16 * 8, doc)
          unicode_trainer(0x1D400, 16 * 16 * 4, doc)
        end
      end
    end
  end
end
