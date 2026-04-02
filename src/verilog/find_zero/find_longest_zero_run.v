module find_longest_zero_run (
    input clk, rst_n,
    input start,                   // Start search
    input [31:0] sram_bm_data_out, // SRAM bitmap data output
    output reg [12:0] start_addr,  // Longest zero run start address
    output reg [12:0] run_length,  // Longest zero run length
    output reg done,               // Search complete
    output reg [7:0] sram_bm_addr, // SRAM bitmap address (256 entries for 8192 bits)
    output reg [31:0] sram_bm_data_in, // SRAM bitmap data input
    output reg sram_bm_ce,         // SRAM bitmap chip enable
    output reg sram_bm_we          // SRAM bitmap write enable
);

    reg [3:0] state, next_state;
    reg [12:0] idx;                // Current index (0 to 8191)
    reg [12:0] curr_len;           // Current zero run length
    reg [12:0] max_len;            // Maximum zero run length
    reg [12:0] max_start;          // Start address of max zero run
    reg [31:0] bitmap_entry;       // Temporary register for SRAM bitmap read
    reg bitmap_bit;                // Specific bit from bitmap entry
    reg [4:0] bit_idx;             // Bit index within 32-bit bitmap entry
    reg [7:0] init_idx;            // Index for SRAM initialization

    localparam IDLE = 4'b0000,
               RESET_SRAM = 4'b0001,
               READ_BIT = 4'b0010,
               CHECK_ZERO = 4'b0011;

    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESET_SRAM;
            idx <= 0;
            curr_len <= 0;
            max_len <= 0;
            max_start <= 0;
            start_addr <= 0;
            run_length <= 0;
            done <= 0;
            bitmap_entry <= 0;
            bitmap_bit <= 0;
            bit_idx <= 0;
            init_idx <= 0;
            sram_bm_addr <= 0;
            sram_bm_data_in <= 0;
            sram_bm_ce <= 0;
            sram_bm_we <= 0;
        end else begin
            state <= next_state;
            case (state)
                RESET_SRAM: begin
                    sram_bm_addr <= init_idx;
                    sram_bm_data_in <= 0;
                    sram_bm_ce <= 1;
                    sram_bm_we <= 1;
                    if (init_idx < 255) begin
                        init_idx <= init_idx + 1;
                    end else begin
                        init_idx <= 0;
                        sram_bm_ce <= 0;
                        sram_bm_we <= 0;
                    end
                end
                IDLE: begin
                    sram_bm_ce <= 0;
                    sram_bm_we <= 0;
                    if (start) begin
                        idx <= 0;
                        curr_len <= 0;
                        max_len <= 0;
                        max_start <= 0;
                        done <= 0;
                        bit_idx <= 0;
                    end
                end
                READ_BIT: begin
                    sram_bm_ce <= 0;
                    sram_bm_addr <= idx[12:5]; // Map 13-bit idx to 8-bit SRAM address
                    bit_idx <= idx[4:0];       // Bit index within 32-bit word
                    sram_bm_ce <= 1;
                    sram_bm_we <= 0;           // Read bitmap
                end
                CHECK_ZERO: begin
                    sram_bm_ce <= 0;
                    bitmap_entry <= sram_bm_data_out;
                    bitmap_bit <= sram_bm_data_out[bit_idx];
                    idx <= idx + 1;
                    if (sram_bm_data_out[bit_idx] == 0) begin
                        curr_len <= curr_len + 1;
                        if (curr_len == 0) max_start <= idx; // Set start of new zero run
                    end else begin
                        if (curr_len > max_len) begin
                            max_len <= curr_len;
                            max_start <= max_start; // Retain previous max_start
                        end
                        curr_len <= 0;
                    end
                    if (idx >= 8192) begin
                        if (curr_len > max_len) begin
                            max_len <= curr_len;
                            max_start <= max_start; // Retain previous max_start
                        end
                        start_addr <= max_start;
                        run_length <= max_len;
                        done <= 1;
                    end
                end
            endcase
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            RESET_SRAM: begin
                if (init_idx == 255) next_state = IDLE;
            end
            IDLE: if (start) next_state = READ_BIT;
            READ_BIT: next_state = CHECK_ZERO;
            CHECK_ZERO: next_state = (idx >= 8192) ? IDLE : READ_BIT;
            default: next_state = IDLE;
        endcase
    end
endmodule
