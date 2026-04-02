module address_check (
    input clk, rst_n,
    input [13:0] virt_addr, // Virtual address
    input end_cmd_in,       // End command
    output reg [13:0] addr_out, // Output address (physical or virtual)
    output reg lookup_flg_out,  // Lookup flag
    output reg end_cmd_out      // End command output
);
    reg [13:0] base_addr;  // Stored base address (previously v_addr)
    reg [12:0] p_blk;      // Physical block
    reg [12:0] cont;       // Continuity length

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_out <= 0;
            lookup_flg_out <= 0;
            end_cmd_out <= 0;
            base_addr <= 0;
            p_blk <= 0;
            cont <= 0;
        end else begin
            addr_out <= virt_addr; // Default to virtual address if no hit
            lookup_flg_out <= 1;   // Default to lookup needed
            end_cmd_out <= end_cmd_in;

            if (virt_addr >= base_addr && virt_addr <= (base_addr + cont)) begin
                addr_out <= p_blk + (virt_addr - base_addr);
                lookup_flg_out <= 0;
            end
        end
    end
endmodule
