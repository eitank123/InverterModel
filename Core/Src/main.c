/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
// Standard CAN Header for Transmission
CAN_HandleTypeDef hcan1; // STM32F4 often uses hcan1

UART_HandleTypeDef huart2;
/* USER CODE BEGIN PV */
// Define the 4 Inverters: FL(1), FR(2), RL(5), RR(6)
AMK_Inverter_t inverters[4];

CAN_TxHeaderTypeDef TxHeader;
uint32_t TxMailbox;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
void AMK_Init_Inverters(void) {
    // Initialize Node IDs (Standard Formula Student IDs)
    uint8_t ids[4] = {1, 2, 5, 6};

    for(int i=0; i<4; i++) {
        inverters[i].node_id = ids[i];
        inverters[i].rx_id = AMK_BASE_RX + ids[i]; // e.g., 0x184
        inverters[i].tx_id = AMK_BASE_TX + ids[i]; // e.g., 0x284
        
        // --- 1. bDcOn (HV Simulation) ---
        // We simulate HV being present immediately on startup.
        inverters[i].hv_present = 1; 
        
        inverters[i].actuals2.Errors = 0;
        inverters[i].actuals1.status_word = 0x1 << 8; // System ready
        inverters[i].setpoints.control_word = 0;
    }
}

// Logic to emulate the AMK State Machine
void AMK_Update_Logic(AMK_Inverter_t *inv) {
    // Extract Control Word Bits
    // Bit 8: Enable
    // Bit 9: Inverter On
    // Bit 10: Remove Error
    inv->actuals1.status_word |= ((inv->setpoints.control_word >> 9) & 1) << 11;
    uint8_t cmd_inverter_on = (inv->control_word >> 9) & 1;
    uint8_t cmd_remove_error = (inv->control_word >> 10) & 1;

    // Error Handling
    if (cmd_remove_error) {
        inv->error_state = 0;
    }

    // --- CHECK: System Ready (SBM) ---
    // SBM requires High Voltage (hv_present) and No Errors
    if (inv->hv_present && !inv->error_state) {
        inv->SBM_ready = 1;
    } else {
        inv->SBM_ready = 0;
    }

    // --- CHECK: bEnable ---
    // To Enable, we need SBM to be true first, then we look at Bit 8 (cmd_enable)
    if (inv->SBM_ready && cmd_enable) {
        inv->drive_enable = 1;
        
        // --- CHECK: bInverterOn ---
        // To turn Inverter On, we need Enable to be true, then look at Bit 9
        if (cmd_inverter_on) {
            inv->inverter_on = 1;
            
            // --- MIRRORING HAPPENS HERE ---
            // The inverter is now "Magnetized" and processing torque.
            // We mirror the requested torque back to actual torque.
            inv->actual_torque = inv->torque_limit_pos;
            inv->actual_velocity = inv->target_velocity; 
        } else {
            inv->inverter_on = 0;
            inv->actual_torque = 0;
            inv->actual_velocity = 0; 
        }
    } else {
        inv->drive_enable = 0;
        inv->inverter_on = 0;
        inv->actual_torque = 0;
        inv->actual_velocity = 0;
    }
}

void AMK_Send_Feedback(AMK_Inverter_t *inv) {
    uint8_t txData[8];
    
    // Construct Status Word
    inv->status_word = 0;
    
    // Bit 8: Controller Enable (Feedback for bEnable)
    if (inv->drive_enable) inv->status_word |= (1 << 8);  
    
    // Bit 9: Inverter On (Feedback for bInverterOn)
    if (inv->inverter_on)  inv->status_word |= (1 << 9);  
    
    // Bit 10: Error
    if (inv->error_state)  inv->status_word |= (1 << 10); 
    
    // Bit 13: System Ready (Feedback for SBM)
    if (inv->SBM_ready)    inv->status_word |= (1 << 13); 
    
    // Bit 14: Quit DC On (Feedback for bDcOn / HV Present)
    if (inv->hv_present)   inv->status_word |= (1 << 14); 

    // Pack Data (Little Endian)
    // Bytes 0-1: Status Word
    txData[0] = inv->status_word & 0xFF;
    txData[1] = (inv->status_word >> 8) & 0xFF;
    
    // Bytes 2-3: Actual Velocity
    txData[2] = inv->actual_velocity & 0xFF;
    txData[3] = (inv->actual_velocity >> 8) & 0xFF;
    
    // Bytes 4-5: Torque Current (Actual Torque)
    txData[4] = inv->actual_torque & 0xFF;
    txData[5] = (inv->actual_torque >> 8) & 0xFF;
    
    // Bytes 6-7: Magnetizing Current (0)
    txData[6] = 0;
    txData[7] = 0;

    TxHeader.StdId = inv->tx_id;
    TxHeader.RTR = CAN_RTR_DATA;
    TxHeader.IDE = CAN_ID_STD;
    TxHeader.DLC = 8;
    TxHeader.TransmitGlobalTime = DISABLE;

    HAL_CAN_AddTxMessage(&hcan1, &TxHeader, txData, &TxMailbox);
}

// Interrupt Callback for CAN Rx
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hCan) {
    CAN_RxHeaderTypeDef RxHeader;
    uint8_t RxData[8];

    if (HAL_CAN_GetRxMessage(hCan, CAN_RX_FIFO0, &RxHeader, RxData) == HAL_OK) {
        for(int i=0; i<4; i++) {
            if(RxHeader.StdId == inverters[i].rx_id) {
                // Parse Request from VCU
                inverters[i].control_word = (RxData[1] << 8) | RxData[0];
                inverters[i].target_velocity = (RxData[3] << 8) | RxData[2];
                inverters[i].torque_limit_pos = (RxData[5] << 8) | RxData[4];
                inverters[i].torque_limit_neg = (RxData[7] << 8) | RxData[6];
                
                AMK_Update_Logic(&inverters[i]);
                break;
            }
        }
    }
}

void AMK_Setup_Filters(void) {
    CAN_FilterTypeDef sFilterConfig;

    sFilterConfig.FilterBank = 0;
    sFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
    sFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
    sFilterConfig.FilterIdHigh = 0x0000;
    sFilterConfig.FilterIdLow = 0x0000;
    sFilterConfig.FilterMaskIdHigh = 0x0000;
    sFilterConfig.FilterMaskIdLow = 0x0000;
    sFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
    sFilterConfig.FilterActivation = ENABLE;
    sFilterConfig.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan1, &sFilterConfig) != HAL_OK) {
        Error_Handler();
    }
}
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/
  SystemClock_Config(void); // Use CubeMX generated clock config here

  
  /* USER CODE BEGIN 1 */
  

  // Start CAN
  if (HAL_CAN_Start(&hcan1) != HAL_OK) {
      Error_Handler();
  }
  
  // Enable Interrupts
  if (HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK) {
      Error_Handler();
  }

  uint32_t last_sim_tick = 0;

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();
  

  /* USER CODE BEGIN Init */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_CAN1_Init(); // Initialize CAN1
  AMK_Init_Inverters();
  AMK_Setup_Filters();
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    uint32_t current_tick = HAL_GetTick();

      if (current_tick - last_sim_tick >= SIM_STEP_MS) {
          last_sim_tick = current_tick;
          
          for(int i=0; i<4; i++) {
              AMK_Update_Logic(&inverters[i]); 
              AMK_Send_Feedback(&inverters[i]);
          }
          
          // Heartbeat LED (PA5 for F4 Nucleo)
          HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5); 
      }
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
static void MX_CAN1_Init(void)
{
  hcan1.Instance = CAN1;
  hcan1.Init.Prescaler = 6; // Adjust based on your APB1 Clock to get 500kbps or 1Mbps
  hcan1.Init.Mode = CAN_MODE_NORMAL;
  hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan1.Init.TimeSeg1 = CAN_BS1_11TQ; // Adjust timing as needed
  hcan1.Init.TimeSeg2 = CAN_BS2_2TQ;
  hcan1.Init.TimeTriggeredMode = DISABLE;
  hcan1.Init.AutoBusOff = DISABLE;
  hcan1.Init.AutoWakeUp = DISABLE;
  hcan1.Init.AutoRetransmission = DISABLE;
  hcan1.Init.ReceiveFifoLocked = DISABLE;
  hcan1.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pin : PA5 (Nucleo LED) */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
