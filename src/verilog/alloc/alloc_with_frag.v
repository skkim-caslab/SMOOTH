module alloc_with_frag (
    input clk, rst_n,
    input [13:0] virt_addr,        // Virtual address
    input [12:0] size,             // Requested block size
    input [2:0] use_cnt,           // Use count
    input [12:0] start_addr,       // Longest zero run start address
    input [12:0] run_length,       // Longest zero run length
    input start_alloc,             // Start allocation
    input [28:0] sram_bt_data_out, // SRAM block table data output
    input [31:0] sram_bm_data_out, // SRAM bitmap data output
    output reg [12:0] p_blk,       // Physical block address (first fragment)
    output reg [12:0] cont,        // Continuity (first fragment)
    output reg done,               // Allocation complete
    output reg [13:0] sram_bt_addr, // SRAM block table address
    output reg [28:0] sram_bt_data_in, // SRAM block table data input
    output reg sram_bt_ce,         // SRAM block table chip enable
    output reg sram_bt_we,         // SRAM block table write enable
    output reg [7:0] sram_bm_addr, // SRAM bitmap address (256 entries for 8192 bits)
    output reg [31:0] sram_bm_data_in, // SRAM bitmap data input
    output reg sram_bm_ce,         // SRAM bitmap chip enable
    output reg sram_bm_we          // SRAM bitmap write enable
);

    reg [3:0] state, next_state;
    reg [13:0] counter;            // Counter for updates
    reg [13:0] base_addr;          // Current virtual address
    reg [12:0] alloc_size;         // Size to allocate
    reg [12:0] remaining_blocks;   // Remaining blocks to allocate
    reg [31:0] bitmap_entry;       // Temporary register for bitmap read
    reg [4:0] bit_idx;             // Bit index within 32-bit bitmap entry
    reg [13:0] init_idx;           // Index for SRAM initialization
    reg [13:0] temp_addr;          // Temporary register for start_addr + counter

    localparam IDLE = 4'b0000,
               RESET_SRAM = 4'b0001,
               CHECK_SIZE = 4'b0010,
               BITMAP_READ = 4'b0011,
               BITMAP_WRITE = 4'b0100,
               TABLE_WRITE = 4'b0101;

    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESET_SRAM;
            counter <= 0;
            base_addr <= 0;
            alloc_size <= 0;
            remaining_blocks <= 0;
            p_blk <= 0;
            cont <= 0;
            done <= 0;
            bitmap_entry <= 0;
            bit_idx <= 0;
            init_idx <= 0;
            temp_addr <= 0;
            sram_bt_addr <= 0;
            sram_bt_data_in <= 0;
            sram_bt_ce <= 0;
            sram_bt_we <= 0;
            sram_bm_addr <= 0;
            sram_bm_data_in <= 0;
            sram_bm_ce <= 0;
            sram_bm_we <= 0;
        end else begin
            state <= next_state;
            case (state)
                RESET_SRAM: begin
                    sram_bt_addr <= init_idx;
                    sram_bt_data_in <= 0;
                    sram_bt_ce <= 1;
                    sram_bt_we <= 1;
                    sram_bm_addr <= init_idx[7:0]; // Initialize bitmap SRAM (256 entries)
                    sram_bm_data_in <= 0;
                    sram_bm_ce <= 1;
                    sram_bm_we <= 1;
                    if (init_idx < 16383) begin
                        init_idx <= init_idx + 1;
                    end else begin
                        init_idx <= 0;
                        sram_bt_ce <= 0;
                        sram_bt_we <= 0;
                        sram_bm_ce <= 0;
                        sram_bm_we <= 0;
                    end
                end
                IDLE: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                    sram_bm_ce <= 0;
                    sram_bm_we <= 0;
                    if (start_alloc && size != 0 && run_length != 0) begin
                        base_addr <= virt_addr;
                        remaining_blocks <= size;
                        alloc_size <= (run_length < size) ? run_length : size;
                        p_blk <= start_addr;
                        cont <= (run_length < size) ? run_length : size;
                        counter <= 0;
                        done <= 0;
                    end
                end
                CHECK_SIZE: begin
                    counter <= 0;
                    temp_addr <= start_addr + counter; // Compute address
                    sram_bm_addr <= temp_addr[12:5];   // Use temp_addr for bit select
                    bit_idx <= temp_addr[4:0];         // Use temp_addr for bit index
                    sram_bm_ce <= 1;
                    sram_bm_we <= 0; // Read bitmap
                end
                BITMAP_READ: begin
                    sram_bm_ce <= 0;
                    bitmap_entry <= sram_bm_data_out;
                    temp_addr <= start_addr + counter; // Compute address for next iteration
                    sram_bm_addr <= temp_addr[12:5];   // Use temp_addr for bit select
                    bit_idx <= temp_addr[4:0];         // Use temp_addr for bit index
                    sram_bm_ce <= 1;
                    sram_bm_we <= 0; // Read bitmap for write preparation
                end
                BITMAP_WRITE: begin
                    sram_bm_ce <= 0;
                    sram_bm_we <= 0;
                    if (counter < alloc_size) begin
                        sram_bm_data_in <= sram_bm_data_out | (32'b1 << bit_idx); // Set bit to 1
                        sram_bm_ce <= 1;
                        sram_bm_we <= 1; // Write to bitmap
                        sram_bt_addr <= base_addr + counter;
                        sram_bt_data_in <= {start_addr + counter, alloc_size - counter, use_cnt};
                        sram_bt_ce <= 1;
                        sram_bt_we <= 1; // Write to block table
                        counter <= counter + 1;
                    end
                end
                TABLE_WRITE: begin
                    sram_bt_ce <= 0;
                    sram_bt_we <= 0;
                    sram_bm_ce <= 0;
                    sram_bm_we <= 0;
                    if (counter >= alloc_size) begin
                        remaining_blocks <= remaining_blocks - alloc_size;
                        base_addr <= base_addr + alloc_size;
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
                if (init_idx == 16383) next_state = IDLE;
            end
            IDLE: if (start_alloc && size != 0 && run_length != 0) next_state = CHECK_SIZE;
            CHECK_SIZE: next_state = BITMAP_READ;
            BITMAP_READ: next_state = BITMAP_WRITE;
            BITMAP_WRITE: begin
                if (counter >= alloc_size) next_state = TABLE_WRITE;
                else next_state = BITMAP_READ;
            end
            TABLE_WRITE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
endmodule
