/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"
#include "stm32f4xx_hal_can.h"
#include "stm32f4xx_hal_uart.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */
typedef struct {
    uint16_t status_word;
    int16_t actual_velocity;
    int16_t torque_current;
    uint16_t magnetizing_current;
} actual_values1;

typedef struct {
    uint16_t Motor_temperature;
    int16_t Plate_temperature;
    int16_t Errors;
} actual_values2;

typedef struct {
    uint16_t control_word;
    int16_t target_velocity;
    int16_t torque_limit_pos;
    int16_t torque_limit_neg;
} setpoint_values;

typedef struct {
    // Configuration
    uint8_t node_id;
    uint32_t rx_id; // Setpoints ID (0x183 + NodeID)
    uint32_t tx_id; // Actual Values ID (0x283 + NodeID)

    uint8_t hv_present;   // bDcOn

    // Outputs (To VCU - Mirroring Inputs)
    actual_values1 actuals1;
    actual_values2 actuals2;
    setpoint_values setpoints;

} AMK_Inverter_t;
/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/

/* USER CODE BEGIN Private defines */
#define AMK_BASE_RX 0x183
#define AMK_BASE_TX 0x283
#define SIM_STEP_MS 10    // Send feedback every 10ms


void AMK_Init_Inverters(void);
void AMK_Update_Logic(AMK_Inverter_t *inv);
void AMK_Send_Feedback(AMK_Inverter_t *inv);
void AMK_Setup_Filters(void);
static void MX_CAN1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_GPIO_Init(void);

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

