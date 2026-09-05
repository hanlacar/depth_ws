# LiDAR extension point (미구현)

향후 별도 LiDAR workspace는 obstacle distance와 collision-free detour path만 제공한다.
`drive=3.00`에서 1.0 m 이내 감속, 0.5 m 이내 정지, S자 우회 후 map route 재합류,
주차 경로 실행은 그 workspace와 MCU의 책임이다. depth stack은 case/localization/mission 상태만 제공한다.
production launch에는 합성 또는 가짜 LiDAR publisher가 없다.
