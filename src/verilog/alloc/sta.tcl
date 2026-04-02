read_liberty ../libs/asap7.lib
read_verilog synthesized_alloc_with_frag.v
link_design alloc_with_frag
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/energy/verilog/alloc.out
report_power >> ../../../data/energy/verilog/alloc.out
