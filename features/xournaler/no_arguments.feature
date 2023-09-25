Feature: xournaler script

 Be helpful for first time users.

 Scenario: no arguments given
   When xournaler is ran
   And the output is read
   Then the output contains: Usage:
   And the output contains: xournaler command [options...]
   And the output contains: Commands: help, info, pages, features, label-map, print-labels, train, scan, training-doc
   And the exit code is 0
