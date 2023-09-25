Feature: xournaler info command

 Display information about Xournal files.

 Scenario: help needed
   Given the command lines:
     | info --help |
     | info -h |
   When xournaler is ran
   And the output is read
   Then the output contains "Usage"
   And the exit code is 0

  Scenario: no arguments
   Given the command line: info
   When xournaler is ran
   And the output is read
   Then the output is empty
   And the exit code is 0

 Scenario: good file
   Given the command line: info features/xournaler/good.xopp
   When xournaler is ran
   And the output is read
   Then the output is:
     |  Path: features/xournaler/good.xopp |
     | Title: Xournal++ document - see https://github.com/xournalpp/xournalpp |
     |  Size: 33164 bytes |
     | Pages: 5 |
     |   0: 612x792, 2 layers, 89 strokes, 81 texts, 0 images |
     |   1: 612x792, 2 layers, 107 strokes, 80 texts, 0 images |
     |   2: 612x792, 2 layers, 72 strokes, 80 texts, 0 images |
     |   3: 612x792, 2 layers, 72 strokes, 80 texts, 0 images |
     |   4: 612x792, 2 layers, 72 strokes, 68 texts, 0 images |
   And the exit code is 0

  Scenario: bad file
   Given the command line: info features/xournaler/info.feature
   When xournaler is ran
   And the error output is read
   Then the error output contains "ParseError"
   And the exit code is 1

  Scenario: no file
   Given the command line: info features/xournaler/notfound
   When xournaler is ran
   And the error output is read
   Then the error output contains "ENOENT"
   And the exit code is 1

  Scenario: many files
   Given the command line: info features/xournaler/good.xopp features/xournaler/good1.xopp
   When xournaler is ran
   And the output is read
   And the error output is read
   Then the output contains "good.xopp"
   And the output is:
     |  Path: features/xournaler/good.xopp |
     | Title: Xournal++ document - see https://github.com/xournalpp/xournalpp |
     |  Size: 33164 bytes |
     | Pages: 5 |
     |   0: 612x792, 2 layers, 89 strokes, 81 texts, 0 images |
     |   1: 612x792, 2 layers, 107 strokes, 80 texts, 0 images |
     |   2: 612x792, 2 layers, 72 strokes, 80 texts, 0 images |
     |   3: 612x792, 2 layers, 72 strokes, 80 texts, 0 images |
     |   4: 612x792, 2 layers, 72 strokes, 68 texts, 0 images |
   Then the error output contains "good1.xopp (Errno::ENOENT)"
   And the exit code is 1
