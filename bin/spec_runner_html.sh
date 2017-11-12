#!/bin/zsh

INDEX=doc/spec/index.html

# Start the index.html
cat <<EOT > $INDEX
<html>
	<head>
		<style>.PASSED { color: green; } .FAILED { color: red; }</style>
	</head>
	<body>
		<ul>
EOT

# The spec runner
function run_spec()
{
  local spec="$1"
  local DIR=`dirname $spec`
  local BASE=`basename -s .spec $spec`
  echo -n "$spec "

  mkdir -p doc/$DIR && bundle exec rspec -Ilib -Iexamples $spec -f html > doc/$DIR/$BASE.html && STATE=PASSED || STATE=FAILED

  echo $STATE

  cat <<-EOT >> $INDEX
	<li class="$STATE"><a href="../$DIR/$BASE.html">$DIR/$BASE</a> &mdash; <span class="state">$STATE</span></li>
EOT
}

# And the loop
for spec in `find spec -name \*.spec | sort`; do
  run_spec "$spec"
done

# Finalize the index
cat <<EOT >> $INDEX
		</ul>
	</body>
</html>
EOT
