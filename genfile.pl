#!/usr/bin/perl
use strict;

my @letters = split('','abcdefghijklmnopqrstuvwxyz0123456789-');
my @exts = ('.com','.net','.org','.ru');

sub get_rand_line
	{
	my $len = int(rand(30)) + 3;
	my $name = $letters[int(rand(36))];
	$name .= $letters[int(rand(37))] for (1 .. $len - 1);
	$name .= $exts[int(rand(scalar(@exts)))];
	my $vlen = int(rand(50));
	my $val = '';
	$val .= $letters[int(rand(36))] for (1..$vlen);
	return ($name,$val);
	}
	
my $count = $ARGV[0];
my ($file1,$file2) = ($ARGV[1],$ARGV[2]);
my ($fh1,$fh2);
if ($file1)
	{ open $fh1,'>',$file1; }
if ($file2)
	{ open $fh2,'>',$file2; }

my %present = ();
for (0 .. $count - 1)
	{
	my ($name,$val) = get_rand_line();
	print $fh1 $name."\t".$val."\n";
	$present{$name} = $name;
	}
	
my $cnt = 0;
while ($cnt < $count)
	{
	my ($name,$val) = get_rand_line();
	next if ($present{$name});
	$present{$name} = 1;
	print $fh2 $name."\t".$val."\n";
	$cnt++;
	}

close ($fh1);
close ($fh2);
	
